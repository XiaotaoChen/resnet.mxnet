


'''
delya_quant: after delay_quant iters, the quantization working actually.
ema_decay:  the hyperparameter for activation threshold update.
grad_mode:  the mode for gradients pass. there are two mode: ste or clip. 
            ste mean straightforward pass the out gradients to data,
            clip mean only pass the gradients whose value of data in the range of [-threshold, threshold],
                        the gradients of outer is settting to 0.
workspace:  the temporary space used in grad_mode=clip
'''
def quantization_int8_quant_attrs(): return {
    "weight_quant_attrs": {
        "delay_quant": "0", 
        "ema_decay": "0.99",
        "grad_mode": "clip",
        "is_weight": "True",
        "is_weight_perchannel": "False",
        "quant_mode": "minmax"
    }, 
    "act_quant_attrs": {
        "delay_quant": "0", 
        "ema_decay": "0.99",
        "grad_mode": "clip",
        "is_weight": "False",
        "is_weight_perchannel": "False",
        "quant_mode": "minmax"
    }
}

def QIL_quant_attrs(nbits): return {
    "weight_quant_attrs": {
        "fix_gamma": "True", 
        "nbits": "4",
        "is_weight": "True"
    }, 
    "act_quant_attrs": {
        "fix_gamma": "True", 
        "nbits": "4",
        "is_weight": "False"
    }
}

def PACT_quant_attrs(nbits): return { 
    "weight_quant_attrs": {
        "nbits": str(nbits)
    },
    "act_quant_attrs": {
        "nbits": str(nbits)
    }
}

def WNQ_quant_attrs(nbits): return { 
    "weight_quant_attrs": {
        "nbits": str(nbits),
        "is_perchannel": "True"
    }, 
    "act_quant_attrs": { }
}

def GDRQ_quant_attrs(nbits): return { 
    "weight_quant_attrs": {
        "nbits": str(nbits),
        "group_size": "-1",
        "is_weight": "True",
        "lamda": "0.01"
    }, 
    # "act_quant_attrs": {
    #     "nbits": "2",
    #     "group_size": "-1",
    #     "is_weight": "False",
    #     "lamda": "0.01"
    # }
    "act_quant_attrs": {
        "nbits": str(nbits),
        "threshold": "8.0"
    }
}


quantize_attrs = {"Quantization_int8" : quantization_int8_quant_attrs, 
                  "QIL": QIL_quant_attrs, 
                  "QIL_V2": QIL_quant_attrs,
                  "QIL_V3": QIL_quant_attrs,
                  "PACT": PACT_quant_attrs,
                  "WNQ": WNQ_quant_attrs,
                  "GDRQ": GDRQ_quant_attrs}

def get_quantize_attrs(quantize_alg, nbits):
    assert quantize_alg in ("Quantization_int8", "QIL", "QIL_V2", "QIL_V3", "PACT", "WNQ", "GDRQ"), "{} not suported".format(quantize_alg)
    attrs = quantize_attrs[quantize_alg](nbits)
    return attrs