/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   dataloader.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 4:01 PM
 */

#ifndef DATALOADER_H
#define DATALOADER_H
#include "ann/xtensor_lib.h"
#include "ann/dataset.h"

using namespace std;

template<typename DType, typename LType>
class DataLoader{
public:
    
private:
    Dataset<DType, LType>* ptr_dataset;
    int batch_size;
    bool shuffle;
    bool drop_last;
    /*
    TODO: add more member variables to support the iteration 
    */
    xt::xarray<unsigned long> indices;  // Used for shuffling and puting Dataset into Batches
    size_t numOfBatch;
    size_t count_batch;
    int m_seed;
public:
    DataLoader(Dataset<DType, LType>* ptr_dataset,
            int batch_size,
            bool shuffle=true,
            bool drop_last=false,
            int seed = -1){
        /*
        TODO: Add your code to do the initialization
        */
        this->ptr_dataset = ptr_dataset;
        this->batch_size = batch_size;
        this->shuffle = shuffle;
        this->drop_last = drop_last;
        this->m_seed = seed;

        size_t numOfElement = ptr_dataset->get_data_shape()[0];  // This will return the number of elements in the 1st dimension
        indices = xt::arange<size_t>(numOfElement);  // indices will be a 1-D array without the elements from the data,
                                    // this means it will be an empty array, used to store the indices of the 1st dimension in data
        indices = xt::arange<unsigned long>(numOfElement);  // Fill in indices with numbers represent the indices of elements in data
        
        // Shuffle the indices of elements in data (only the 1st dimentsion)
        if (shuffle) {
            if (seed >= 0) {
                xt::random::seed(this->m_seed);  // Use the same seed for reproducibility
            }
            xt::random::shuffle(indices);
        }

        numOfBatch = numOfElement / batch_size;
        count_batch = 0;
    }
    virtual ~DataLoader(){}
    
    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// START: Section                                                     //
    /////////////////////////////////////////////////////////////////////////
    
    /*TODO: Add your code here to support iteration on batch*/

    /* 
    *Your implementation must support the following syntax
    DataLoader<double, double>* pLoader; //initialized
    ///Hidden code 1
    for(auto batch: *pLoader){  // The *pLoader dereferences the pointer might need to use the operator * in class Iterator
        xt::xarray<double> X = batch.getData();  // ?This get method belongs to the class Batch
        xt::xarray<double> t = batch.getLabel();  // ?This get method belongs to the class Batch
        ///Hidden code 2
    }
    ///Hidden code 3
    */

    ////////// BATCH: BEGIN //////////
    // This method will be called each iteration.
    // For example, in the code teacher provide, the for loop 
    // for ( auto batch : loader )
    // each iteration will call this method and the batch returned will assign to batch
    Batch<DType, LType> getBatch(size_t batchStart) {
        size_t cap = ptr_dataset->get_label_shape()[0];  // This make sure the batch end doesn't exceed the number of elements in data 
        size_t batchEnd = std::min(batchStart + batch_size, cap);  // If batchStart + batch_size exceeds, take the cap
        size_t current_batch_size = batchEnd - batchStart;

        // To include the last elements
        if (!drop_last && cap % batch_size != 0 && (batchStart == (numOfBatch * batch_size) - batch_size)) {
            current_batch_size += cap % batch_size; 
            batchEnd += cap % batch_size;
        }

        // Get the shape of the batchData and batchLabel
        // First, we copy the get_data_shape() and get_data_shape() to 2 other svectors to copy the elements in them
        // to shapeData and shapeLabel and change from unsigned long to size_t
        // Second, we copy the size of those svector to shapeData and shapeLabel
        // Third, we manually copy each element in uns_long_shapeData and uns_long_shapeLabel since the svectors
        // can't use the xt::cast
        // (1)
        xt::svector<unsigned long> uns_long_shapeData = ptr_dataset->get_data_shape();
        xt::svector<unsigned long> uns_long_shapeLabel = ptr_dataset->get_data_shape();  // !Problem: 
                                // if get_label_shape(): terminate called after throwing an instance of 'std::bad_array_new_length'
                                // if get_data_shape(): run but failed at data28, 29, 30
        // (2)
        xt::svector<size_t> shapeData(uns_long_shapeData.size()); 
        xt::svector<size_t> shapeLabel(uns_long_shapeLabel.size());
        // (3)
        for (size_t i = 0; i < uns_long_shapeData.size(); ++i) {
            shapeData[i] = static_cast<size_t>(uns_long_shapeData[i]);
        }
        for (size_t i = 0; i < uns_long_shapeLabel.size(); ++i) {
            shapeLabel[i] = static_cast<size_t>(uns_long_shapeLabel[i]);
        }

        if (ptr_dataset->get_data_shape().size() == 1) {  // If the data is a 1-dimensional array
            shapeData = {current_batch_size};
        } else {  // If the data is an n-dimensional array (n > 1)
            shapeData[0] = current_batch_size;
        }
        
        if (ptr_dataset->get_label_shape().size() == 1) {  // If the label is a 1-dimensional array
            shapeLabel = {current_batch_size};
        } else {  // If the label is an n-dimensional array (n > 1)
            shapeLabel[0] = current_batch_size;
        }

        // Create 2 arrays to store the data and the label
        auto shape = ptr_dataset->get_label_shape();  // Get the shape of the label
        xt::xarray<DType> batchData(shapeData);
        xt::xarray<LType> batchLabel(shapeLabel);
        if (ptr_dataset->get_label_shape().size() != 1) {  // If the label is empty
            xt::xarray<LType> batchLabel_empty;
            // For-loop to fetch the data and labels into the batch
            for (size_t i = 0; i < current_batch_size; i++) {
                size_t index = indices[batchStart + i];
                DataLabel<DType, LType> data_and_label = ptr_dataset->getitem(index);

                xt::view(batchData, i, xt::all()) = data_and_label.getData();
            }

            return Batch<DType, LType>(batchData, batchLabel_empty);
        }
        
        // For-loop to fetch the data and labels into the batch
        for (size_t i = 0; i < current_batch_size; i++) {
            size_t index = indices[batchStart + i];
            DataLabel<DType, LType> data_and_label = ptr_dataset->getitem(index);

            xt::view(batchData, i, xt::all()) = data_and_label.getData();
            xt::view(batchLabel, i, xt::all()) = xt::view(data_and_label.getLabel(), xt::all());
        }

        return Batch<DType, LType>(batchData, batchLabel);
    }
    ////////// -BATCH: END //////////

    ////////// ITERATOR METHODS: BEGIN //////////
    class Iterator {
    private:
        DataLoader* pLoader;
        size_t current_batch_start;

    public:
        Iterator(DataLoader* pLoader, size_t current_batch_start) {
            this->pLoader = pLoader;
            this->current_batch_start = current_batch_start;
        }
        
        Batch<DType, LType> operator*() {
            return pLoader->getBatch(current_batch_start);   
        }
        // Prefix
        Iterator& operator++() {
            current_batch_start += pLoader->batch_size;  // To move on to the next batch
            pLoader->count_batch++;
            if (!pLoader->drop_last && pLoader->numOfElements() % pLoader->batch_size != 0
                    && pLoader->count_batch == pLoader->numOfBatch) {
                current_batch_start += pLoader->batch_size;
            }
            return *this;
        }
        // Postfix
        Iterator operator++(int) {
          // TODO implement
            Iterator iterator = *this;
            current_batch_start += pLoader->batch_size;  // To move on to the next batch
            pLoader->count_batch++;
            if (!pLoader->drop_last && pLoader->numOfElements() % pLoader->batch_size != 0
                    && pLoader->count_batch == pLoader->numOfBatch) {
                current_batch_start += pLoader->batch_size;
            }
            return iterator;
        }

        bool operator!=(const Iterator& other) {
            return this->current_batch_start != other.current_batch_start;  // To compare 2 batches base on their current_batch_start
        }
    };

    size_t numOfElements() {
        return ptr_dataset->get_data_shape()[0];
    }

    Iterator begin() {
        count_batch = 0;
        if (numOfBatch == 0) {  // If there is no batch, then the iterator will not process, therefore no batch will be fetched
            return end();
        }
        return Iterator(this, 0);  // The current_batch_start will be initialize with 0
    }
    Iterator end() {
        if (numOfBatch == 0) { // If there is no batch, then the iterator will not process, therefore no batch will be fetched
            return Iterator(this, 0);
        }
        size_t endIndex = ptr_dataset->get_data_shape()[0];  // This is also the number of samples (the number of element in the
                                                            // 1st dimension of data)
        if (drop_last && endIndex % batch_size != 0) {
            endIndex = (endIndex / batch_size) * batch_size;  // This will only take the integer part
        } else if (!drop_last && endIndex % batch_size != 0) {
            endIndex = ((endIndex / batch_size) * batch_size) + batch_size;  // This will only take the integer part
        }

        return Iterator(this, endIndex);
    }
    ////////// -ITERATOR METHODS: END //////////
    
    /////////////////////////////////////////////////////////////////////////
    // The section for supporting the iteration and for-each to DataLoader //
    /// END: Section                                                       //
    /////////////////////////////////////////////////////////////////////////
};


#endif /* DATALOADER_H */

