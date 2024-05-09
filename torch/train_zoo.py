import tqdm


class Train:

    def __init__(self,
                 generative_model,
                 num_of_epochs,
                 loss_function,
                 optimizer) -> None:
        """
        Parameters
        -------
        generative_model: nn.Module
            nn that generates the image
        num_of_epochs: int
            numper of epochs
        loss_functin: nn.Module
            loss fucntion
        optimizer: torch.optim
            optimizer
        """
        self.model = generative_model
        self.epochs = int(num_of_epochs)
        self.batchs = self.epochs//100
        self.optimizer = optimizer
        self.loss_fn = loss_function

    def train_step(self):
        """one train step"""
        train_loss = []
        for i in tqdm.tqdm(range(self.epochs)):
            self.optimizer.zero_grad()
            img = self.model()
            loss = self.loss_fn(img)
            loss.backward()
            self.optimizer.step()
            if i % self.batchs == 0:
                train_loss.append(loss)
        return train_loss

    def test_step(self, loss, i):
        """one test step"""
        self.model.eval()
        self.train_loss.append(loss)
        if i % self.output_freq == 0:
            print(loss)

    def train(self):
        """implementation of train/test steps
        on each epoch"""
        train_loss = []
        for i in tqdm.tqdm(range(self.epochs)):
            loss = self.train_step()
            self.test_step(loss, i)
        return train_loss
