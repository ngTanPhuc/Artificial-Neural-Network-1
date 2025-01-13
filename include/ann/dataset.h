/*
 * Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
 * Click nbfs://nbhost/SystemFileSystem/Templates/cppFiles/file.h to edit this template
 */

/* 
 * File:   dataset.h
 * Author: ltsach
 *
 * Created on September 2, 2024, 3:59 PM
 */

#ifndef DATASET_H
#define DATASET_H
#include "ann/xtensor_lib.h"
using namespace std;

template<typename DType, typename LType>
class DataLabel{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
public:
    DataLabel(xt::xarray<DType> data,  xt::xarray<LType> label):
    data(data), label(label){
    }
    xt::xarray<DType> getData() const{ return data; }
    xt::xarray<LType> getLabel() const{ return label; }
};

template<typename DType, typename LType>
class Batch{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
public:
    Batch(xt::xarray<DType> data,  xt::xarray<LType> label):
    data(data), label(label){
    }
    virtual ~Batch(){}
    xt::xarray<DType>& getData(){return data; }
    xt::xarray<LType>& getLabel(){return label; }
};


template<typename DType, typename LType>
class Dataset{
private:
public:
    Dataset(){};
    virtual ~Dataset(){};
    
    virtual int len()=0;
    virtual DataLabel<DType, LType> getitem(int index)=0;
    virtual xt::svector<unsigned long> get_data_shape()=0;
    virtual xt::svector<unsigned long> get_label_shape()=0;
    
};

//////////////////////////////////////////////////////////////////////
template<typename DType, typename LType>
class TensorDataset: public Dataset<DType, LType>{
private:
    xt::xarray<DType> data;
    xt::xarray<LType> label;
    xt::svector<unsigned long> data_shape, label_shape;
    
public:
    /* TensorDataset: 
     * need to initialize:
     * 1. data, label;
     * 2. data_shape, label_shape
    */
    TensorDataset(xt::xarray<DType> data, xt::xarray<LType> label){
        /* 
        TODO: your code is here for the initialization
         */
        this->data = data;
        this->label = label;

                            // Type-casting        // Iterate through the container (here, the container is svector)
        this->data_shape = xt::svector<unsigned long>(data.shape().begin(), data.shape().end());
        this->label_shape = xt::svector<unsigned long>(label.shape().begin(), label.shape().end());
    }
    /* len():
     *  return the size of dimension 0
    */
    int len(){
        /* 
        TODO: your code is here to return the dataset's length
         */
        return static_cast<int>(this->data.shape()[0]); // The .shape() method returns a vector, therefore,
                                                        // to get the size of the 1st dimension means to get
                                                        // to get the first element in this vector
    }
    
    /* getitem:
     * return the data item (of type: DataLabel) that is specified by index
     */
    DataLabel<DType, LType> getitem(int index){
        /* 
        TODO: your code is here
         */
        if (index < 0 || index >= this->len()) {
            throw out_of_range("Index is out of range!");
        }
        
        xt::xarray<DType> returnData;
        xt::xarray<LType> returnLabel;
        if (data.dimension() != 0) {
            if (data.shape().size() == 1) {  // If the data is a 1-dimensional array
                returnData = xt::view(this->data, index);
            } else {  // If the data is an n-dimensional array (n > 1)
                returnData = xt::view(this->data, index, xt::all());
            }

            if (label.dimension() == 0) {  // Case 1: Data is available but labels are not (is empty or is a scalar)
                returnLabel = this->label;
            } else {  // Case 2: Data and labels are both available
                if (label.shape().size() == 1) {  // If the label is a 1-dimensional array
                    returnLabel = xt::view(this->label, index);
                } else {  // If the label is an n-dimensional array (n > 1)
                    returnLabel = xt::view(this->label, index, xt::all());
                }
            }
        }

        return DataLabel<DType, LType>(returnData, returnLabel);
    }
    
    xt::svector<unsigned long> get_data_shape(){
        /* 
        TODO: your code is here to return data_shape
         */
        return data_shape;
    }
    xt::svector<unsigned long> get_label_shape(){
        /* 
        TODO: your code is here to return label_shape
         */
        return label_shape;
    }
};

#endif /* DATASET_H */

