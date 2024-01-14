from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch 
import torchvision
import os
import torchvision.transforms
from PIL import Image
import torchvision
from zmq import device
from models.hvae import HVAE
from models.vae import VAE
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    def __init__(self, tensors, labels):
        self.tensors = tensors
        self.labels = labels

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        return self.tensors[idx], self.labels[idx]
    
def load_images_from_directory(directory_path, transform):
    images = []
    for filename in os.listdir(directory_path):
        if filename.endswith(".png"):
            image_path = os.path.join(directory_path, filename)
            image = Image.open(image_path)
            image = transform(image)
            images.append(image)
    return images

def load_data(train_data_0, train_data_1, test_data_0, test_data_1, val_data_0, val_data_1, batch_size):
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), 
                                            torchvision.transforms.Grayscale(),
                                            torchvision.transforms.Resize((256,256)),
                                            torchvision.transforms.Normalize(0, 1)
                                            ])
    
    images_0_train = load_images_from_directory(train_data_0, transform)
    images_1_train = load_images_from_directory(train_data_1, transform)
    images_0_test = load_images_from_directory(test_data_0, transform)
    images_1_test = load_images_from_directory(test_data_1, transform)
    images_0_val = load_images_from_directory(val_data_0, transform)
    images_1_val = load_images_from_directory(val_data_1, transform)

    train_0_dataset = CustomDataset(images_0_train, torch.zeros(len(images_0_train)))
    train_1_dataset = CustomDataset(images_1_train, torch.ones(len(images_1_train)))
    test_0_dataset = CustomDataset(images_0_test, torch.zeros(len(images_0_test)))
    test_1_dataset = CustomDataset(images_1_test, torch.ones(len(images_1_test)))
    val_0_dataset = CustomDataset(images_0_val, torch.zeros(len(images_0_val)))
    val_1_dataset = CustomDataset(images_1_val, torch.ones(len(images_1_val)))

    train_full_dataset = ConcatDataset([train_0_dataset, train_1_dataset])
    test_full_dataset = ConcatDataset([test_0_dataset, test_1_dataset])
    val_full_dataset = ConcatDataset([val_0_dataset, val_1_dataset])

    # Create data loaders
    train_loader = DataLoader(train_full_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_full_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_full_dataset, batch_size=batch_size, shuffle=True)


    return train_loader, test_loader, val_loader

def get_project_data():
    train_path_0 = r'./archive/train/0'
    train_path_1 = r'./archive/train/1'
    test_path_0 = r'./archive/test/0'
    test_path_1 = r'./archive/test/1'
    val_path_0 = r'./archive/val/0'
    val_path_1 = r'./archive/val/1'

    train_loader, test_loader, val_loader = load_data(train_path_0, train_path_1, test_path_0, test_path_1, val_path_0, val_path_1, batch_size=100)
    print("dataloader created")
    return train_loader, test_loader, val_loader

def train_hvae(model: HVAE, train_loader, optim, device):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    model.to(device)
    for x, y in train_loader:
        image, label = x, y
        optim.zero_grad()   
        pred, mu_1, logvar_1, mu_2, logvar_2, mu1_star, logvar_1_star = model(image.to(device))
        
        recon_loss, kld = model.elbo_loss(image.to(device), pred, mu_1, logvar_1, mu_2, logvar_2, mu1_star, logvar_1_star)
        loss = recon_loss + kld
        loss.backward()
        optim.step()

        total_loss += loss.cpu().data.numpy()*image.shape[0]
        reconstruction_loss += recon_loss.cpu().data.numpy()*image.shape[0]
        kld_loss += kld.cpu().data.numpy()*image.shape[0]
    
    reconstruction_loss /= len(train_loader.dataset)
    kld_loss /= len(train_loader.dataset)
    total_loss /= len(train_loader.dataset)

    return total_loss, kld_loss, reconstruction_loss

def test_hvae(epoch, model: HVAE, test_loader, device, path_for_images):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    model.to(device)
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            image, label = x
            pred, mu_1, logvar_1, mu_2, logvar_2, mu1_star, logvar_1_star = model(image.to(device))
            recon_loss, kld = model.elbo_loss(image.to(device), pred, mu_1, logvar_1, mu_2, logvar_2, mu1_star, logvar_1_star)
            loss = recon_loss + kld

            total_loss += loss.cpu().data.numpy()*image.shape[0]
            reconstruction_loss += recon_loss.cpu().data.numpy()*image.shape[0]
            kld_loss += kld.cpu().data.numpy()*image.shape[0]
            if i == 0:
                plot(epoch, pred.cpu().data.numpy(), label, path_for_images)

    reconstruction_loss /= len(test_loader.dataset)
    kld_loss /= len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, kld_loss,reconstruction_loss  


def clear_dir(dir_path):
    
    if not os.path.isdir(dir_path):
        return
    
    files = os.listdir(dir_path)

    for file in files:
        file_path = os.path.join(dir_path, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def plot(epoch, pred, y, path):
    if not os.path.isdir(path):
        os.mkdir(path)
    fig = plt.figure(figsize=(16,16))
    for i in range(6):
        ax = fig.add_subplot(3,2,i+1)
        ax.imshow(pred[i,0],cmap='gray')
        ax.axis('off')
        ax.set_title(str(y[i]))
    plt.savefig(f"{path}/epoch_{epoch}.jpg")
    plt.close()

def train_vae(model: VAE, train_loader, optim, device):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    model.to(device)
    for x, y in train_loader:
        image, label = x, y

        optim.zero_grad()   
        pred, mu_1, logvar_1 = model(image.to(device))
        recon_loss, kld = model.elbo_loss(image.to(device), pred, mu_1, logvar_1)
        loss = recon_loss + kld
        loss.backward()
        optim.step()

        total_loss += loss.cpu().data.numpy()*image.shape[0]
        reconstruction_loss += recon_loss.cpu().data.numpy()*image.shape[0]
        kld_loss += kld.cpu().data.numpy()*image.shape[0]
    
    reconstruction_loss /= len(train_loader.dataset)
    kld_loss /= len(train_loader.dataset)
    total_loss /= len(train_loader.dataset)

    return total_loss, kld_loss, reconstruction_loss

def test_vae(epoch, model: VAE, test_loader, device, path_for_images):
    reconstruction_loss = 0
    kld_loss = 0
    total_loss = 0
    model.to(device)
    with torch.no_grad():
        for i, x in enumerate(test_loader):
            image, label = x
            pred, mu_1, logvar_1 = model(image.to(device))
            recon_loss, kld = model.elbo_loss(image.to(device), pred, mu_1, logvar_1)
            loss = recon_loss + kld

            total_loss += loss.cpu().data.numpy()*image.shape[0]
            reconstruction_loss += recon_loss.cpu().data.numpy()*image.shape[0]
            kld_loss += kld.cpu().data.numpy()*image.shape[0]
            if i == 0:
                plot(epoch, pred.cpu().data.numpy(), label, path_for_images)

    reconstruction_loss /= len(test_loader.dataset)
    kld_loss /= len(test_loader.dataset)
    total_loss /= len(test_loader.dataset)
    return total_loss, kld_loss,reconstruction_loss  

def only_encode_vae(model: VAE, val_loader, device):
    model.to(device)
    points = []
    labels = []
    with torch.no_grad():
        for images, label in val_loader:
            z = model.only_encode(images.to(device))
            points.append(z)
            labels.append(label)
    return points, labels

def decode_latent_vae(model:VAE, latent, device):
    model.to(device)
    with torch.no_grad():
        pred = model.only_decode(latent.to(device))
        return pred

def perturb_dim_vae(latent, device, model, value, path):

    latent = latent.cpu().numpy()

    batch_size = 10

    for i in range(0, 128, batch_size):
        updated_latent = latent
        updated_latent[:, i:i+batch_size] += value
        updated_latent = torch.tensor(latent)
        decoded_image = decode_latent_vae(model, updated_latent, device).cpu()
        if not os.path.isdir(path):
            os.mkdir(path)
        plt.imshow(torch.reshape(decoded_image, (256,256)), cmap='gray')
        plt.axis('off')
        plt.savefig(f"{path}/dim_{i}.jpg")
        plt.close()

    return
