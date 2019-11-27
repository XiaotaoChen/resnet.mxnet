"""
-----------------------------------------
example of quantize setting
-----------------------------------------
Quantization_int8
==============================
    "weight":{
        "quantize_op_name": "Quantization_int8",
        "init_value": 0,
        "attrs": {
            "nbits": "4",
            "quant_mode": "minmax",
            "is_weight": "True",
            "is_weight_perchannel": "False",
            "delay_quant": "0",
            "ema_decay": "0.99",
            "grad_mode": "ste",
            "fix_act_scale": "False"
        }
    },
    "act":{
        "quantize_op_name": "Quantization_int8",
        "init_value": 0,
        "attrs": {
            "nbits": "4",
            "quant_mode": "minmax",
            "is_weight": "False",
            "is_weight_perchannel": "False",
            "delay_quant": "0",
            "ema_decay": "0.99",
            "grad_mode": "ste",
            "fix_act_scale": "False"
        }
    }

GDRQ
==============================
    "weight":{
        "quantize_op_name": "GDRQ_CXX",
        "init_value": 0.5,
        "attrs": {
            "nbits": "4",
            "fix_alpha": "False",
            "group_size": "-1",
            "is_weight": "True",
            "lamda": "0.001",
            "delay_quant": "0",
            "ktimes": "3",
            "weight_grad_mode": "ste"
        }
    },
    "act":{
        "quantize_op_name": "GDRQ_CXX",
        "init_value": 1.0,
        "attrs": {
            "nbits": "4",
            "fix_alpha": "True",
            "group_size": "-1",
            "is_weight": "False",
            "lamda": "0.001",
            "delay_quant": "0",
            "ktimes": "3"
        }
    }

PACT
==============================
    "weight":{
        "quantize_op_name": "DoReFa_CXX",
        "attrs": {
            "nbits": "4"
        }
    },
    "act":{
        "quantize_op_name": "PACT_CXX",
        "init_value": 8.0,
        "attrs": {
            "nbits": "4"
        }
    }


LSQ
==============================
    "weight":{
        "quantize_op_name": "LSQ_PY",
        "init_value": 3.0,
        "attrs": {
            "nbits": "4",
            "is_weight": "True",
            "grad_factor": "False"
        }
    },
    "act":{
        "quantize_op_name": "LSQ_PY",
        "init_value": 8.0,
        "attrs": {
            "nbits": "4",
            "is_weight": "False",
            "grad_factor": "False"
        }
    }

COLL_INFO
==============================
    "weight":{
        "quantize_op_name": "COLL_INFO_PY",
        "attrs": {
            "is_weight": "True"
        }
    },
    "act":{
        "quantize_op_name": "COLL_INFO_PY"
        "attrs": {
            "is_weight": "False"
        }
    }

"""