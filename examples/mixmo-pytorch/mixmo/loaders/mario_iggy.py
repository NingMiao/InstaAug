import numpy as np
import torch
import torch.nn.functional as F


class MarioIggy(torch.utils.data.TensorDataset):
    def __init__(self, path, train=True, **kwargs):
        # TODO: Need to avoid generating twicecthe dataset (for train=True and False)
        (train_imgs, train_labels), (test_imgs, test_labels) = self.generate_mario_data(
            path, **kwargs
        )
        if train:
            imgs, labels = train_imgs, train_labels
        else:
            imgs, labels = test_imgs, test_labels
        super().__init__(imgs, labels)

    @staticmethod
    def generate_mario_data(
        path, n_train, n_test, max_rot_angle=np.pi / 2.0, dataseed=0
    ):
        imgs = np.load(path + "mario_iggy.npz")
        mario = torch.FloatTensor(imgs["mario"])
        iggy = torch.FloatTensor(imgs["iggy"])
        
        ##Change 1x3x32x32 to 1x3x64x64
        mario_new=torch.zeros([1,3,64, 64]).type(mario.type())
        mario_new[:,:,::2,::2]=mario
        mario_new[:,:,1::2,::2]=mario
        mario_new[:,:,::2,1::2]=mario
        mario_new[:,:,1::2,1::2]=mario
        mario=mario_new
        iggy_new=torch.zeros([1,3,64, 64]).type(mario.type())
        iggy_new[:,:,::2,::2]=iggy
        iggy_new[:,:,1::2,::2]=iggy
        iggy_new[:,:,::2,1::2]=iggy
        iggy_new[:,:,1::2,1::2]=iggy
        iggy=iggy_new

        n_train_each = int(n_train / 2)
        n_test_each = int(n_test / 2)

        train_mario = torch.cat(n_train_each * [mario])
        train_iggy = torch.cat(n_train_each * [iggy])

        test_mario = torch.cat(n_test_each * [mario])
        test_iggy = torch.cat(n_test_each * [iggy])

        torch.random.manual_seed(dataseed)

        train_mario_up_angles = (
            torch.rand(int(n_train_each / 2)) * max_rot_angle - max_rot_angle / 2
        )  # angles for training mario upwards
        train_mario_dn_angles = (
            torch.rand(int(n_train_each / 2)) * max_rot_angle
            - max_rot_angle / 2
            + np.pi
        )  # angles for training mario downwards
        train_iggy_up_angles = (
            torch.rand(int(n_train_each / 2)) * max_rot_angle - max_rot_angle / 2
        )  # angles for training iggy upwards
        train_iggy_dn_angles = (
            torch.rand(int(n_train_each / 2)) * max_rot_angle
            - max_rot_angle / 2
            + np.pi
        )  # angles for training iggy downwards
        train_angles = torch.cat(
            [
                train_mario_up_angles,
                train_mario_dn_angles,
                train_iggy_up_angles,
                train_iggy_dn_angles,
            ],
            dim=0,
        )

        
        if True:
            l=int(n_test_each / 2)
            r=torch.tensor(np.arange(l))/l
            test_mario_up_angles = (
                r * max_rot_angle - max_rot_angle / 2
            )  # angles for testing mario upwards
            test_mario_dn_angles = (
                r * max_rot_angle - max_rot_angle / 2 + np.pi
            )  # angles for testing mario downwards
            test_iggy_up_angles = (
                r * max_rot_angle - max_rot_angle / 2
            )  # angles for testing iggy upwards
            test_iggy_dn_angles = (
                r * max_rot_angle - max_rot_angle / 2 + np.pi
            )  # angles for testing iggy downwards
        else:
            test_mario_up_angles = (
                torch.rand(int(n_test_each / 2)) * max_rot_angle - max_rot_angle / 2
            )  # angles for testing mario upwards
            test_mario_dn_angles = (
                torch.rand(int(n_test_each / 2)) * max_rot_angle - max_rot_angle / 2 + np.pi
            )  # angles for testing mario downwards
            test_iggy_up_angles = (
                torch.rand(int(n_test_each / 2)) * max_rot_angle - max_rot_angle / 2
            )  # angles for testing iggy upwards
            test_iggy_dn_angles = (
                torch.rand(int(n_test_each / 2)) * max_rot_angle - max_rot_angle / 2 + np.pi
            )  # angles for testing iggy downwards
        
        test_angles = torch.cat(
            [
                test_mario_up_angles,
                test_mario_dn_angles,
                test_iggy_up_angles,
                test_iggy_dn_angles,
            ],
            dim=0,
        )

        train_labs = torch.cat(
            [
                0 * torch.ones([int(n_train_each / 2)]),
                1 * torch.ones([int(n_train_each / 2)]),
                2 * torch.ones([int(n_train_each / 2)]),
                3 * torch.ones([int(n_train_each / 2)]),
            ],
            dim=0,
        ).to(dtype=torch.long)
        test_labs = torch.cat(
            [
                0 * torch.ones([int(n_test_each / 2)]),
                1 * torch.ones([int(n_test_each / 2)]),
                2 * torch.ones([int(n_test_each / 2)]),
                3 * torch.ones([int(n_test_each / 2)]),
            ],
            dim=0,
        ).to(dtype=torch.long)
        
        ## combine to just train and test ##
        train_images = torch.cat((train_mario, train_iggy))
        test_images = torch.cat((test_mario, test_iggy))

        ## rotate ##
        # train #
        with torch.no_grad():
            # Build affine matrices for random translation of each image
            affineMatrices = torch.zeros(n_train, 2, 3)
            affineMatrices[:, 0, 0] = train_angles.cos()
            affineMatrices[:, 1, 1] = train_angles.cos()
            affineMatrices[:, 0, 1] = train_angles.sin()
            affineMatrices[:, 1, 0] = -train_angles.sin()

            flowgrid = F.affine_grid(
                affineMatrices, size=train_images.size(), align_corners=True
            )
            train_images = F.grid_sample(train_images, flowgrid, align_corners=True)

        # test #
        with torch.no_grad():
            # Build affine matrices for random translation of each image
            affineMatrices = torch.zeros(n_test, 2, 3)
            affineMatrices[:, 0, 0] = test_angles.cos()
            affineMatrices[:, 1, 1] = test_angles.cos()
            affineMatrices[:, 0, 1] = test_angles.sin()
            affineMatrices[:, 1, 0] = -test_angles.sin()

            flowgrid = F.affine_grid(
                affineMatrices, size=test_images.size(), align_corners=True
            )
            test_images = F.grid_sample(test_images, flowgrid, align_corners=True)

        ## shuffle ##
        trainshuffler = np.random.permutation(n_train)
        
        if True:
            testshuffler = np.arange(n_test)
        else:
            testshuffler = np.random.permutation(n_test)

        train_images = train_images[np.ix_(trainshuffler), ::].squeeze()
        train_labs = train_labs[np.ix_(trainshuffler)]

        test_images = test_images[np.ix_(testshuffler), ::].squeeze()
        
        test_labs = test_labs[np.ix_(testshuffler)]
        

        if False:
            np.save('tem/mario_train_image.npy', train_images.detach().cpu().numpy())
            np.save('tem/mario_train_label.npy', train_labs.detach().cpu().numpy())
            np.save('tem/mario_test_image.npy', test_images.detach().cpu().numpy())
            np.save('tem/mario_test_label.npy', test_labs.detach().cpu().numpy())
        
        return (train_images, train_labs), (test_images, test_labs)
