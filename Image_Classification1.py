from __future__ import print_function
import argparse
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.optim as optim 
from torchvision import datasets, transforms 
from torch.optim.lr_scheduler import StepLR

class Net ( nn.Module ):
    def __init__(self):
        super ( Net , self ).__init__()
        self.conv1Layer = nn.Conv2d( 1 , 32 , 3 , 1 )
        self.conv2Layer = nn.Conv2d( 32 , 64 , 3 , 1)

        self.dropout1Layer = nn.Dropout( 0.25 )
        self.dropout2Layer = nn.Dropout( 0.5 )
        
        self.fc1Layer = nn.Linear( 9216 , 128 )
        self.fc2Layer = nn.Linear( 128 , 10 ) 

    def forward ( self , k ):
        k = self.conv1Layer( k )
        k = F.relu( k )
        k = self.conv2Layer( k )
        k = F.relu( k )

        k = F.max_pool2d( k , 2 )
        k = self.dropout1Layer( k )
        k = torch.flatten( k , 1 ) 
        k = self.fc1Layer( k )
        k = F.relu( k )

        x = self.dropout2Layer( k )
        x = self.fc2Layer( k )
        
        output = F.log_softmax( k , dim = 1 )
        return output

def train( args , model , device , train_loader , optimizer , epoch ):
    model.train() 

    for batch_idx, ( data , target ) in enumerate( train_loader ):

        data, target = data.to( device ) , target.to( device )
        optimizer.zero_grad()

        output = model( data ) 
        loss = F.nll_loss( output , target )

        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:

            print( 'Train Epoch: {} [{}/{} ( {:.0f} % ) ] \ tLoss: {:.6f}'.format ( epoch, batch_idx * len( data ), len( train_loader.dataset ) ,
                100. * batch_idx / len( train_loader ) , loss.item() ) )
            
            if args.dry_run:
                break


def test( model, device, test_loader ):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad(): #we dont want to update the parameters of the model, so this keeps everything in place

        for data, target in test_loader:
            data, target = data.to( device ) , target.to( device )
            output = model( data )

            test_loss += F.nll_loss( output , target , reduction = 'sum' ).item()  # sum up batch loss
            pred = output.argmax( dim = 1 , keepdim = True )  # get the index of the max log-probability
            correct += pred.eq( target.view_as( pred ) ).sum().item()
            test_loss /= len( test_loader.dataset )

    print('\nTest set: Average loss: {:.4f} , Accuracy: {}/{} ( {:.0f} % ) \n'.format ( test_loss, correct , len( test_loader.dataset ),
        100. * correct / len( test_loader.dataset ) ) )


def main():

    # Training settings
    parser = argparse.ArgumentParser( description = 'PyTorch MNIST Example' )

    parser.add_argument( '--batch-size' , type = int , default = 64 , metavar = 'N' , help = 'input batch size for training ( default: 64 )' )
    parser.add_argument( '--test-batch-size' , type = int , default = 1000 , metavar = 'N' , help = 'input batch size for testing ( default: 1000 )' )
    parser.add_argument( '--epochs' , type = int , default = 1 , metavar = 'N' , help = 'number of epochs to train ( default: 5 )' )
    parser.add_argument( '--lr' , type = float , default = 1.0 , metavar = 'LR' , help =' learning rate ( default: 1.0 )' )
    parser.add_argument( '--gamma' , type = float , default = 0.7 , metavar = 'M' , help = 'Learning rate step gamma ( default: 0.7 )' )
    parser.add_argument( '--no-cuda' , action = 'store_true' , default = False , help = 'disables CUDA training' )
    parser.add_argument( '--no-mps' , action = 'store_true' , default = False , help = 'disables macOS GPU training' )
    parser.add_argument( '--dry-run', action = 'store_true' , default = False , help = 'quickly check a single pass' )
    parser.add_argument( '--seed', type = int , default = 1 , metavar = 'S' , help = 'random seed (default: 1)' ) #seed = to fix the random (if dont fix the seed, every time you train the model, it will get several different outputs)(like how the accuracy increases each time you run it, if the seed is fixed, it will not increase and it will stay the same
    parser.add_argument( '--log-interval' , type = int , default = 10 , metavar = 'N' , help = 'how many batches to wait before logging training status' ) 
    parser.add_argument( '--save-model', action = 'store_true' , default = False , help = 'For Saving the current Model' )
    
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed( args.seed )

    if use_cuda:
        device = torch.device( "cuda" )
    elif use_mps:
        device = torch.device( "mps" )
    else:
        device = torch.device( "cpu" ) 

    train_kwargs = { 'batch_size': args.batch_size } #kwargs = parameter (here, line 74 we defined the batch_size)
    test_kwargs = { 'batch_size': args.test_batch_size } 

    if use_cuda:

        cuda_kwargs = { 'num_workers': 1 , 'pin_memory': True , 'shuffle': True }

        train_kwargs.update( cuda_kwargs )
        test_kwargs.update( cuda_kwargs )

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize( ( 0.1307, ) , ( 0.3081, ) ) 
        ]) 
    
    dataset1 = datasets.MNIST( '../data' , train = True , download = True , transform = transform )#they defined the training dataset!! (can see this in the parameters) (training has 50,000 images)
                       
    dataset2 = datasets.MNIST('../data' , train = False , transform = transform ) #defined as the testing dataset!! (the test data is smaller than the training data) (like our example, testing has 10,000 images)
                       
    train_loader = torch.utils.data.DataLoader( dataset1 , **train_kwargs ) #load the training data to the train_loader
    test_loader = torch.utils.data.DataLoader( dataset2 , **test_kwargs ) #load the testing data to the test_loader

    model = Net().to( device ) #define the model 
    optimizer = optim.Adadelta( model.parameters() , lr = args.lr ) #defines the optimizer (they chose to use the Adadelta optimizer (similar to SGD) which is rarely used)

    scheduler = StepLR( optimizer,  step_size = 1 , gamma = args.gamma ) #scheduler defines the Learning Rate 

    for epoch in range( 1 , args.epochs + 1 ): #from line 133-136, they define the training step and the test step

        train( args , model , device , train_loader , optimizer , epoch )  
        test( model , device , test_loader )
        scheduler.step() 

    if args.save_model: 

        torch.save( model.state_dict() , "mnist_cnn.pt" )

if __name__ == '__main__': 
    main()