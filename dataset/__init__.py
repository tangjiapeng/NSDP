from dataset.dataset_deform4d_flow import Deform4DFlow_Dataset
from dataset.dataset_deformtransfer_flow import DeformTransferFlow_Dataset
from dataset.dataset_userhandle_flow import DeformUserhandle_Dataset

dataset_dict = {
    'deform4d': Deform4DFlow_Dataset,
    'deformtransfer': DeformTransferFlow_Dataset,
    'tosca':    DeformUserhandle_Dataset,
    'dogrec':   DeformUserhandle_Dataset,
}