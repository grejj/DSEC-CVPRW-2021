'''
Predictor loads data and does the training and testing
'''

from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.provider import DatasetProvider
from dataset.visualization import disp_img_to_rgb_img, show_disp_overlay, show_image
from estimator import Network

class Predictor(object):
    
    def __init__(self, dataset_path: Path, delta_t_ms: int=50, num_bins=15):
        self.dataset_path = dataset_path
        print("Dataset path={}".format(self.dataset_path))

        self.dsec_dir = Path(self.dataset_path)
        assert self.dsec_dir.is_dir()

        self.train_path = self.dsec_dir / 'train'
        print("Train path={}".format(self.train_path))
        assert self.dsec_dir.is_dir(), str(self.dataset_path)
        assert self.train_path.is_dir(), str(self.train_path)

        self.delta_t_ms = delta_t_ms
        self.num_bins = num_bins

        self.dataset_provider = DatasetProvider(dataset_path=self.dsec_dir, num_bins=1)

    def visualize(self):
        batch_size = 1
        num_workers = 0
        train_loader = DataLoader(
                dataset=self.get_train_dataset(),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                drop_last=False)
        with torch.no_grad():
            for data in tqdm(train_loader):
                if batch_size == 1:
                    disp = data['disparity_gt'].numpy().squeeze()
                    disp_img = disp_img_to_rgb_img(disp)
                    #if args.overlay:
                    left_voxel_grid = data['representation']['left'].squeeze()
                    ev_img = torch.sum(left_voxel_grid, axis=0).numpy()
                    ev_img = (ev_img/ev_img.max()*256).astype('uint8')
                    show_disp_overlay(ev_img, disp_img, height=480, width=640)
                    #else:
                    #    show_image(disp_img)

    def train(self):
        train_loader = DataLoader(
                dataset=self.get_train_dataset(),
                batch_size=1,
                shuffle=True,
                num_workers=2)

        network = Network()
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.RMSprop(params=network.parameters(), lr=0.01)

        for epoch in range(1000): # loop over dataset mutliple times
            running_loss = 0.0
            for i, data in enumerate(train_loader, 0):

                # parse data

                # torch tensor shape = [1, 480, 640]
                label = data['disparity_gt'].squeeze()

                # torch tensor shape = [1]
                file_index = data['file_index']
                # dict() with keys { 'left' , 'right' }
                representation = data['representation']

                # torch tensors shape = [1, 15, 480, 640] 
                left = representation['left']
                right = representation['right']
                
                # torch tensor shape = [2, 15, 480, 640] 
                inputs = torch.cat((left,right), 1)

                # print(labels[0].size())
                # print(type(file_index))
                # print(type(representation))
                # print(labels.size())
                # print(file_index.size())
                # print(representation.keys())
                # print(type(left))
                # print(type(right))
                print(left.squeeze().size())
                print(right.squeeze().size())
                # print(inputs.size())

                # zero parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                print("input shape: {}".format(inputs.size()))
                outputs = network(inputs)
                print("output shape: {}".format(outputs.size()))
                loss = criterion(outputs.squeeze(), label)
                loss.backward()
                optimizer.step()

                show_image(disp_img_to_rgb_img(outputs.squeeze().detach().numpy()))

                # print statistics
                # running_loss += loss.item()
                # if i % 2000 == 1999: # every 2000
                #     print("[{},{}] loss:{}".format(epoch + 1, i + 1, running_loss / 2000))
                #     running_loss = 0.0
                print("[{},{}] loss:{}".format(epoch + 1, i + 1, loss.item()))

        print("Done training.")
        torch.save(network.state_dict(), './dsec_net.pth')

    def get_train_dataset(self):
        return self.dataset_provider.get_train_dataset()

    def get_val_dataset(self):
        # Implement this according to your needs.
        raise NotImplementedError

    def get_test_dataset(self):
        # Implement this according to your needs.
        raise NotImplementedError

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dsec_dir', help='Path to DSEC dataset directory')
    parser.add_argument('--visualize', action='store_true', help='Visualize data')
    parser.add_argument('--train', action='store_true', help='Train model')
    args = parser.parse_args()

    predictor = Predictor(args.dsec_dir)

    if args.visualize:
        predictor.visualize()

    if args.train:
        predictor.train()