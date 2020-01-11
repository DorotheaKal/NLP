import matplotlib.pyplot as plt 
def plot_loss(l_train,l_test,n_epochs):
    
    plt.figure(figsize = (8,8))
    epochs = [e for e in  range(n_epochs)]
    plt.plot(l_train,epochs,label = 'Train Set Loss',color = 'r')
    plt.plot(l_test,epochs,label = 'Test Set Loss', color = 'm')
    plt.grid()
    plt.legend()
    plt.title('Train and Test loss per Epoch',fontsize = 20)
    plt.ylabel('Loss',fontsize = 18)
    plt.xlabel('Epoch',fontsize = 18)
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
