"""
-----------------------------------------
example of quantize setting
-----------------------------------------
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
            "ktimes": "3"
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

"""