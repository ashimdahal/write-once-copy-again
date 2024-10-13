import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, DistributedSampler
from torchvision import datasets, transforms
import torch.cuda.amp as amp

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from convkan import ConvKAN, LayerNorm2D

import argparse
import os
from datetime import timedelta
import numpy as np 

from sklearn.model_selection import train_test_split

os.environ['NCCL_BLOCKING_WAIT'] = '0'
# torch.cuda.empty_cache()
def setup():
    dist.init_process_group(backend='nccl', timeout=timedelta(seconds=7200000))
    # torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()


class AlexNetKAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            # nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=2),
            # nn.ReLU(inplace=True),
            ConvKAN(3, 32, padding=2, kernel_size=11, stride=4),
            LayerNorm2D(32),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.Conv2d(64, 192, kernel_size=5, padding=2),
            # nn.ReLU(inplace=True),
            ConvKAN(32, 96, kernel_size=5, padding=2),
            LayerNorm2D(96),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            ConvKAN(96, 172, kernel_size=3, padding=1),
            LayerNorm2D(172),
            # nn.Conv2d(384, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            ConvKAN(172, 128, kernel_size=3, padding=1),
            LayerNorm2D(128),
            # nn.Conv2d(256, 256, kernel_size=3, padding=1),
            # nn.ReLU(inplace=True),
            ConvKAN(128, 128, kernel_size=3, padding=1),
            LayerNorm2D(128),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(128 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1000),
        )

        self.flat = nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        # x = x.view(x.size(0), 128*6*6)
        x = self.flat(x)
        x = self.classifier(x)
        return x


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
        accumulation_steps :int = 2
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.valid_data = valid_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path

        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.loss_fn = nn.CrossEntropyLoss()
        self.hist = []

        self.accumulation_steps = accumulation_steps
        self.current_step= 0
        self.scaler = amp.GradScaler()
        
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot()


    def _load_snapshot(self):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(self.snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot['MODEL_STATE'])
        self.epochs_run = snapshot["EPOCHS_RUN"] 
        self.hist = snapshot["HIST"]
        print(f"Snapshot loaded at epoch : {self.epochs_run}")

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.state_dict(),
            "EPOCHS_RUN": epoch,
            "HIST": self.hist
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

    def _run_batch_and_get_loss(self, source, targets):
        with amp.autocast():
            loss = self.get_loss((source, targets))
            loss = loss / self.accumulation_steps

        self.scaler.scale(loss).backward()
        torch.cuda.synchronize()

        if (self.current_step + 1) % (self.accumulation_steps) == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        self.current_step+=1
        return loss

    def _run_epoch_and_get_loss(self, epoch):
        self.model.train()
        self.current_step = 0
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch}"
                f"| Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        train_loss = []
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch_and_get_loss(source, targets)
            train_loss.append(loss)

        return train_loss

    @staticmethod
    def get_accuracy(labels, preds):
        preds = torch.argmax(preds, dim=1)
        acc = (labels==preds).sum()/len(labels)
        return acc

    def get_loss(self, batch):
        features,labels = batch
        preds = self.model(features)
        loss = self.loss_fn(preds, labels)
        return loss

    def validate(self, batch ):
        feature, labels = batch
        loss = self.get_loss(batch)
        pred = self.model(feature)
        acc = self.get_accuracy(labels, pred)
        return {'valid_loss' : loss , 'valid_acc' : acc}
    
    def average_validation(self, out):
        loss = torch.stack([l['valid_loss'] for l in out]).mean()
        acc = torch.stack([l['valid_acc'] for l in out]).mean()
        return {'valid_loss': loss.item() , 'valid_acc': acc.item()}

    @torch.no_grad()
    def validate_and_get_metrics(self):
        self.model.eval()
        out = []
        for source, targets in self.valid_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            out.append(self.validate((source, targets)))
        return self.average_validation(out)

    @staticmethod
    def log_epoch( e, epoch, res):
        
        print('[{} / {}] epoch/s, training loss is {:.4f} validation loss is {:.4f}, validation accuracy is {:.4f} '\
              .format(e+1,epoch,
                      res['train_loss'],
                      res['valid_loss'],                
                      res['valid_acc']
                     )
             )

    def train(self, max_epochs: int):
        for epoch in range(self.epochs_run, max_epochs):
            train_loss = self._run_epoch_and_get_loss(epoch)
            log_dict = self.validate_and_get_metrics()
            log_dict['train_loss'] = torch.stack(train_loss).mean().item()

            if self.gpu_id == 0 and epoch % self.save_every == 0:
                self._save_snapshot(epoch)

            if self.gpu_id == 0:
                self.hist.append(log_dict)
                self.log_epoch(epoch, max_epochs, log_dict)


def load_data():
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_ds = datasets.ImageNet(
                            root="./imagenet/train/", 
                            split='train', 
                            transform=transform,
                        )

    val_ds = datasets.ImageNet(
                            root="./imagenet/val/", 
                            split='val', 
                            transform=transform,
                        )
    # targets = train_ds.targets
    # _, test_train = train_test_split(np.arange(len(targets)), test_size = 0.05, stratify = targets, random_state=42 )
    # train_ds = Subset(train_ds, test_train)
    #
    # valid_targets = val_ds.targets
    # _, test_val = train_test_split(np.arange(len(valid_targets)), test_size = 0.05, stratify = valid_targets, random_state=42)
    # val_ds = Subset(val_ds, test_val)

    print(f" testing dataset size: {len(train_ds)}")
    print(f" testing validation size: {len(val_ds)}")

    return train_ds, val_ds

def prepare_data(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size= batch_size,
        # num_workers=0,
        shuffle=False,
        pin_memory=True,
        sampler=DistributedSampler(dataset)
    )

def load_model():
    model = AlexNetKAN()
    opt = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
    return model, opt

def trainer_agent(epochs:int, save_every:int, snapshot_path:str):
    setup()
    train_data, val_data = load_data()

    batch_size = 16 
    train_dl = prepare_data(train_data, batch_size)
    val_dl = prepare_data(val_data, batch_size * 2)

    model, opt = load_model()

    trainer = Trainer(
                model,
                train_dl,
                val_dl,
                opt,
                save_every,
                snapshot_path
    )
    trainer.train(epochs)
    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--save', type=int, help="frequency to save")
    parser.add_argument("-p","--path",help="Path to store snapshot")
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs")
    args = parser.parse_args()

    trainer_agent(args.epochs, args.save, args.path)

if __name__ == "__main__":
    import time 
    start = time.time()
    main()
    end = time.time()
    print(f"time taken to train CNN: {end-start}")

