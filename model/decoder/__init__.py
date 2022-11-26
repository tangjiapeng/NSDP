from model.decoder.crosstransformer_decoder import CrossTransformerDecoder
from model.decoder.interpolation_decoder import PointInterpDecoder


decoder_dict = {
    'interp': PointInterpDecoder,
    'crossatten': CrossTransformerDecoder,
}