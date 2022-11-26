from model.encoder.pointnetplusplus import PointNetPlusPlusEncoder
from model.encoder.pointransformer import PointTransformerEncoder

encoder_dict = {
    'pointnet++': PointNetPlusPlusEncoder,
    'pointransformer': PointTransformerEncoder,
}