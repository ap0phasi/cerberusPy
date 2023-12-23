import numpy as np
import pandas as pd

def generate_predictions(model,selected_data):
    gen_call = selected_data['call']
    gen_contexts = [selected_data[key] for key in selected_data if 'context' in key]
    gen_response = np.zeros([1,selected_data['response'].shape[1],selected_data['response'].shape[2]])
    
    respones_generated = []
    for igen in range(gen_response.shape[1]):
        res_out = model.predict(
            [gen_call] + gen_contexts + [gen_response]
        )
        gen_response[:,igen,:] = res_out[0,:]
        
        respones_generated.append(res_out[0,:])
        
    return np.vstack(respones_generated)
        