from abc import abstractmethod
from utils.transforms import transform
import matplotlib.pyplot as plt


class AttackError(Exception):
    pass


class BudgetExhaustionError(AttackError):
    pass


class AttackUnableToCompleteError(AttackError):
    pass



class Attack:
    @abstractmethod
    def __init__(self, model, model_config, attack_config):
        self._model = model
        self.model_config = model_config
        self.attack_config = attack_config
        self.firstLogits=[]
        self.secondLogits = []
        self.plotIndex=[]
        self.currentIndex=0

    @abstractmethod
    def attack_targeted(self, x, y):
        pass

    @abstractmethod
    def attack_untargeted(self, x):
        pass

    def get_cache_hits(self):
        return self._model.cache_hits

    def get_total_queries(self):
        return self._model.total

    def reset(self):
        self._model.reset()

    def _check_budget(self, budget):
        if self.get_total_queries() > budget:
            raise BudgetExhaustionError(
                f'Attack budget exhausted: {self.get_total_queries()} > {budget}')

    def model(self, x):
        if self.attack_config["adaptive"]["query_blinding_transform"] is not None:
            x = transform(x, **self.attack_config["adaptive"]["query_blinding_transform"])
        out = self._model(x)
        self._check_budget(self.attack_config['budget'])
        return out

    def end(self, reason):
        self.plot()
        raise AttackUnableToCompleteError(reason)
    
    def plot(self):
        # Create the plot
        print("Generating plot for attack")
        plt.figure(figsize=(30, 6))
        plt.plot(self.plotIndex, self.firstLogits, marker='o', linestyle='-', color='b', label='First Logit')
        plt.plot(self.plotIndex, self.secondLogits, marker='x', linestyle='-', color='r', label='Second Logit')
        # Add labels and title
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title('2-D Plot of HSJA Attack probings')
        plt.legend()
        # Show the plot
        plt.grid(True)
        plt.show()
        print("Done with plot generation")

    def xprint(*args, **kwargs):
        #print( ""+"".join(map(str,args))+"", **kwargs)
        return

    def updatePlot(self, firstLogit, secondLogit):
        self.firstLogits.append(firstLogit)
        self.secondLogits.append(secondLogit)
        self.plotIndex.append(self.currentIndex)
        self.currentIndex+=1
    
    def getCurrentIndex(self):
        return self.currentIndex