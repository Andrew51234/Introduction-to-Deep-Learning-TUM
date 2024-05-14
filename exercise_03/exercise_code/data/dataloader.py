"""Definition of Dataloader"""

import numpy as np


class DataLoader:
    """
    Dataloader Class
    Defines an iterable batch-sampler over a given dataset
    """
    def __init__(self, dataset, batch_size=1, shuffle=False, drop_last=False):
        """
        :param dataset: dataset from which to load the data
        :param batch_size: how many samples per batch to load
        :param shuffle: set to True to have the data reshuffled at every epoch
        :param drop_last: set to True to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size.
            If False and the size of dataset is not divisible by the batch
            size, then the last batch will be smaller.
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):

        ########################################################################
        # TODO:                                                                #
        # Define an iterable function that samples batches from the dataset.   #
        # Each batch should be a dict containing numpy arrays of length        #
        # batch_size (except for the last batch if drop_last=True)             #
        # Hints:                                                               #
        #   - np.random.permutation(n) can be used to get a list of all        #
        #     numbers from 0 to n-1 in a random order                          #
        #   - To load data efficiently, you should try to load only those      #
        #     samples from the dataset that are needed for the current batch.  #
        #     An easy way to do this is to build a generator with the yield    #
        #     keyword, see https://wiki.python.org/moin/Generators             #
        #   - Have a look at the "DataLoader" notebook first. This function is #
        #     supposed to combine the functions:                               #
        #       - combine_batch_dicts                                          #
        #       - batch_to_numpy                                               #
        #       - build_batch_iterator                                         #
        #     in section 1 of the notebook.                                    #
        ########################################################################
        

#         dataset is the dataset that the dataloader should load.
# batch_size is the mini-batch size, i.e. the number of samples you want to load at a time.
# shuffle is a binary boolean (True or False) and defines whether the dataset should be randomly shuffled or not.
# drop_last: is binary and defines how to handle the last mini-batch in your dataset. Specifically, if the amount of samples in your dataset is not dividable by the mini-batch size, there will be some samples left over in the end. If drop_last=True, we simply discard those samples, otherwise we return them together as a smaller mini-batch.
        
        dataset = self.dataset
        batch_size = self.batch_size
        shuffle = self.shuffle
        drop_last = self.drop_last

        #step 1: Build batches

        batches = []
        batch = []
        
        for i in range(len(dataset)):
            batch.append(dataset[i])
            if(len(batch) == batch_size):
                batches.append(batch)
                batch = []
        # print("batches: ", batches)

        #step 2: Combine batches dicts

        combined_batch_dicts = []
        for batch in batches:
            batch_dict = {}
            for data_dict in batch:
                for key, value in data_dict.items():
                    if key not in batch_dict:
                        batch_dict[key] = []
                    batch_dict[key].append(value)
            combined_batch_dicts.append(batch_dict)
            # print("batch_dict: ", batch_dict)

        # print("combined_batch_dicts: ", combined_batch_dicts)


        #step 3: Convert to numpy arrays

        combined_numpy_batches = []
        batch_numpy = {}
        for batch_dict in combined_batch_dicts:
            for key, value in batch_dict.items():
                batch_numpy[key] = np.array(value)
            combined_numpy_batches.append(batch_numpy)
            batch_numpy = {}
        # print("combined_numpy_batches: ", combined_numpy_batches)

        #step 4: build generator
    #         if shuffle:
    #     index_iterator = iter(np.random.permutation(len(dataset)))  # define indices as iterator
    # else:
    #     index_iterator = iter(range(len(dataset)))  # define indices as iterator

    # batch = []
    # for index in index_iterator:  # iterate over indices using the iterator
    #     batch.append(dataset[index])
    #     if len(batch) == batch_size:
    #         yield batch  # use yield keyword to define a iterable generator
    #         batch = []

        def build_batch_iterator():
            if shuffle:
                index_iterator = iter(np.random.permutation(len(dataset)))
            else:
                index_iterator = iter(range(len(dataset)))

            batch = []
            for index in index_iterator:
                data_dict = dataset[index]
                batch.append(data_dict)
                if len(batch) == batch_size:
                    yield {k: [dic[k] for dic in batch] for k in batch[0]}
                    batch = []

            if batch and not drop_last:  # handle the remaining items when drop_last is False
                yield {k: [dic[k] for dic in batch] for k in batch[0]}

        batch_iterator = build_batch_iterator()
        return batch_iterator

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def __len__(self):
        length = len(self.dataset)
        ########################################################################
        # TODO:                                                                #
        # Return the length of the dataloader                                  #
        # Hint: this is the number of batches you can sample from the dataset. #
        # Don't forget to check for drop last (self.drop_last)!                #
        ########################################################################

        #get the batch size
        batch_size = self.batch_size
        #check if the last batch should be dropped
        drop_last = self.drop_last

        if batch_size == 1:
            length = length
        elif drop_last:
            length = length // batch_size
        else:
            length = length // batch_size + 1

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return length
