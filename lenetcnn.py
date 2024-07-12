import os
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import argparse
# DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def setup():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_data: DataLoader,
        valid_data: DataLoader,
        optimizer: torch.optim.Optimizer,
        save_every: int,
        snapshot_path: str,
    ) -> None:
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.model = model.to(self.gpu_id)
        self.train_data = train_data
        self.valid_data = valid_data
        self.optimizer = optimizer
        self.save_every = save_every
        self.epochs_run = 0
        self.snapshot_path = snapshot_path
        if os.path.exists(snapshot_path):
            print("Loading snapshot")
            self._load_snapshot(snapshot_path)

        self.model = DDP(self.model, device_ids=[self.gpu_id])
        self.loss_fn = nn.CrossEntropyLoss()
        self.hist = []
        
    def _load_snapshot(self, snapshot_path):
        loc = f"cuda:{self.gpu_id}"
        snapshot = torch.load(snapshot_path, map_location=loc)
        self.model.load_state_dict(snapshot["MODEL_STATE"])
        self.epochs_run = snapshot["EPOCHS_RUN"]
        print(f"Resuming training from snapshot at Epoch {self.epochs_run}")

    def _run_batch_and_get_loss(self, source, targets):
        self.optimizer.zero_grad()
        loss = self.get_loss((source, targets))
        loss.backward()
        self.optimizer.step()
        return loss

    def _run_epoch_and_get_loss(self, epoch):
        b_sz = len(next(iter(self.train_data))[0])
        print(f"[GPU{self.gpu_id}] Epoch {epoch} | Batchsize: {b_sz} | Steps: {len(self.train_data)}")
        self.train_data.sampler.set_epoch(epoch)
        train_loss = []
        for source, targets in self.train_data:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch_and_get_loss(source, targets)
            train_loss.append(loss)

        return train_loss

    def _save_snapshot(self, epoch):
        snapshot = {
            "MODEL_STATE": self.model.module.state_dict(),
            "EPOCHS_RUN": epoch,
        }
        torch.save(snapshot, self.snapshot_path)
        print(f"Epoch {epoch} | Training snapshot saved at {self.snapshot_path}")

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
        # acc = accuracy(labels, pred)
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
    my_transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.13,), std=(0.308,))
    ])

    train_ds = datasets.MNIST(root="./mnist/train", train=True, download=False, transform=my_transforms)
    test_ds = datasets.MNIST(root="./mnist/test", train=False, download=False, transform=my_transforms)
    return train_ds, test_ds

def prepare_data(dataset, batch_size):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
        sampler=DistributedSampler(dataset)
    )
    
def load_model():
    model = LeNet5(10)
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    return model, opt

def trainer_agent(save_every:int, snapshot_path:str, epochs:int):
    setup()

    train_ds, test_ds = load_data()

    batch_size = 512 
    train_data = prepare_data(train_ds, batch_size)
    test_data = prepare_data(test_ds,1024)

    model, opt  = load_model()

    trainer = Trainer(model, train_data, test_data, opt, save_every, snapshot_path)
    trainer.train(epochs)

    cleanup()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s','--save', type=int, help="frequency to save")
    parser.add_argument("-p","--path",help="Path to store snapshot")
    parser.add_argument("-e", "--epochs", type=int, help="number of epochs")
    args = parser.parse_args()

    trainer_agent(args.save, args.path, args.epochs)

if __name__ == "__main__":
    main()
