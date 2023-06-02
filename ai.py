import torch
from torch import nn
from torch import optim
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import math

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image)

def save_image(tensor, filename):
    transform = transforms.Compose([
        transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225]),  # unnormalize
        transforms.ToPILImage(),
    ])
    image = transform(tensor)
    image.save(filename)

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features),
            nn.PReLU(),
            nn.Conv2d(in_features, in_features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_features)
        )

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        block4 = []
        for _ in range(upsample_block_num):
            block4.append(nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1))
            block4.append(nn.PixelShuffle(upscale_factor=2))
            block4.append(nn.PReLU())
        self.block4 = nn.Sequential(*block4)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block1 + block3)

        return (torch.tanh(block4) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))


class SRGAN(nn.Module):
    def __init__(self, generator, discriminator):
        super(SRGAN, self).__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.vgg = models.vgg19(pretrained=True).features[:18]  # using the feature maps of the vgg19 network
        self.vgg.eval()  # set VGG to evaluation mode
        for param in self.vgg.parameters():
            param.requires_grad = False

    def perceptual_loss(self, original_image, generated_image):
        original_image = self.vgg(original_image)
        generated_image = self.vgg(generated_image)
        loss = nn.MSELoss()(original_image, generated_image)
        return loss

    def forward(self, x):
        generated_image = self.generator(x)
        disc_output = self.discriminator(generated_image)
        perc_loss = self.perceptual_loss(x, generated_image)
        return generated_image, disc_output, perc_loss


def main():
    # Load the image
    img = load_image('input.jpg')

    # Create the generator and discriminator
    generator = Generator(scale_factor=4)
    discriminator = Discriminator()

    # Create the model and move it to the GPU if available
    model = SRGAN(generator, discriminator)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Move the image to the GPU if available
    img = img.to(device).unsqueeze(0)  # add batch dimension

    # Define the loss function
    adversarial_loss = nn.BCELoss().to(device)

    # Define the optimizer
    optimizer = optim.Adam(model.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training loop
    for epoch in range(num_epochs):
        # Set model to training mode
        model.train()

        # Reset the optimizer gradients
        optimizer.zero_grad()

        # Forward pass
        generated_image, disc_output, perc_loss = model(img)

        # Calculate the generator and discriminator losses
        gen_loss = adversarial_loss(disc_output, torch.ones_like(disc_output))
        dis_loss = adversarial_loss(model.discriminator(generated_image.detach()), torch.zeros_like(disc_output))

        # Add the perceptual loss
        total_loss = gen_loss + dis_loss + perc_loss

        # Backward pass
        total_loss.backward()

        # Update the generator parameters
        optimizer.step()

        # Print the losses for monitoring
        print(f"Epoch [{epoch}/{num_epochs}] Loss: {total_loss.item()}")

    # Remove batch dimension and move tensor back to CPU for saving
    output = generated_image.squeeze(0).cpu()

    # Save the output image
    save_image(output, 'output.jpg')

if __name__ == "__main__":
    num_epochs = 10
    main()
