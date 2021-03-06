import matplotlib.pyplot as plt 
def plot_loss(l_train,l_test,n_epochs,dataset,model_name):
    '''
        plot train and test loss per epoch

    '''
    plt.figure(figsize = (8,8))
    epochs = [e for e in  range(1,n_epochs+1)]
    plt.plot(epochs,l_train,label = 'Train Set Loss',color = 'r')
    plt.plot(epochs,l_test,label = 'Test Set Loss', color = 'm')
    plt.grid()
    plt.legend(fontsize = 15)
    plt.title(f'Learning Curve {dataset}_{model_name}',fontsize = 18)
    plt.ylabel('Loss',fontsize = 18)
    plt.xlabel('Epoch',fontsize = 18)
    plt.xticks(fontsize = 15)

    plt.yticks(fontsize = 15)
    plt.savefig(f'./img/{dataset}/{model_name}_loss.png',dpi = 300)
