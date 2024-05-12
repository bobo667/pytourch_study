import torch


class Nn(torch.nn.Module):

    def __init__(self):
        super(Nn, self).__init__()
        self.cover = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 32, 5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(32, 64, 5, padding=2),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 4 * 4, 64),
            torch.nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.cover(x)
        return x



if __name__ == '__main__':
    nn = Nn()
    input = torch.ones((64, 3, 32, 32))
    output = nn(input)
    print(output.shape)
