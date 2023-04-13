from math import pi 
import torch
from prosailvae.utils import torch_select_unsqueeze

def get_layer_1_neuron_weights(ver="2.1"):
    if ver =="2.1":
        w1 = torch.tensor([- 0.023406878966470, + 0.921655164636366, + 0.135576544080099, - 1.938331472397950, - 3.342495816122680, + 0.902277648009576,
                                + 0.205363538258614, - 0.040607844721716, - 0.083196409727092, + 0.260029270773809, + 0.284761567218845])
        w2 = torch.tensor([- 0.132555480856684, - 0.139574837333540, - 1.014606016898920, - 1.330890038649270, + 0.031730624503341, - 1.433583541317050,
                                - 0.959637898574699, + 1.133115706551000, + 0.216603876541632, + 0.410652303762839, + 0.064760155543506])
        w3 = torch.tensor([+ 0.086015977724868, + 0.616648776881434, + 0.678003876446556, + 0.141102398644968, - 0.096682206883546, - 1.128832638862200,
                                + 0.302189102741375, + 0.434494937299725, - 0.021903699490589, - 0.228492476802263, - 0.039460537589826,]) 
        w4 = torch.tensor([- 0.109366593670404, - 0.071046262972729, + 0.064582411478320, + 2.906325236823160, - 0.673873108979163, - 3.838051868280840,
                                + 1.695979344531530, + 0.046950296081713, - 0.049709652688365, + 0.021829545430994, + 0.057483827104091])
        w5 = torch.tensor([- 0.089939416159969, + 0.175395483106147, - 0.081847329172620, + 2.219895367487790, + 1.713873975136850, + 0.713069186099534,
                                + 0.138970813499201, - 0.060771761518025, + 0.124263341255473, + 0.210086140404351, - 0.183878138700341,])
    elif ver =="3A":    
        w1 = torch.tensor([-1.4710156196971582,-0.2563532709468933,-2.2180860576490495,-1.085898980193225,1.476503732633888,-0.203130406532602,
                           1.9247696053465146,1.836433120470074,-0.4178313281505014,0.11120379207368114,0.2710853575035239])
        w2 = torch.tensor([1.8061903884637975,1.1335141059415899,0.7529844590307209,-1.849059724727423,-2.3218651443747502,-0.14636054606555837,
                           -0.55740658813474,0.6841136954400282,0.8247349654342588,-0.6080497708934546,0.32421327324338656])
        w3 = torch.tensor([0.7970672934944534,0.4460131825143041,0.408586425658519,-1.485087188758423,1.1623034379249386,-1.4153189332758485,
                           1.4595324130059253,0.3335744055337625,-1.6186285460736094,-0.14549546441013367,-2.8578688750531547]) 
        w4 = torch.tensor([0.09421670372484336,-0.22700197694341617,-0.15056963844888643,-1.2842480628443311,0.6965044295491283,1.9782367938424654,
                           -0.8916551406435398,-0.1039817992235624,0.013484692816631437,0.007841478637455265,-0.02581868621411949])
        w5 = torch.tensor([-0.9667116308950019,-0.4183310913759375,-1.890580979987651,0.0066940023538766685,1.9755057593307033,4.045480239195276,
                           -1.3979478080853602,-0.7644964732510922,0.010792479104631214,1.1088706990036719,0.04211359644818833])
    elif ver =="3B":    
        w1 = torch.tensor([0.0830079594419,-0.288281933525,-0.111764138832,-2.25405934742,0.268135370597,3.14923063777,
                           -1.06996677449,-0.1837991652,0.0149841814287,0.0314033988537,0.00514853694825])
        w2 = torch.tensor([1.00387064453,-0.813625826848,0.740886365416,1.00938214456,-0.926530517091,-0.476882469139,
                           -0.0798642487375,-1.01674056491,-0.959418129249,0.749986744207,-1.2989372502])
        w3 = torch.tensor([-0.914903852701,0.418794470829,0.156396260112,0.320562039933,2.0695705545,1.6938367995,
                           -0.307752980774,0.092487209186,-2.07129204917,-1.00442058831,-0.531933380552]) 
        w4 = torch.tensor([-0.000281553791956,0.250188709989,-0.151786460873,-0.8446482236,-1.02084725541,0.00997687410719,
                           -0.151504370026,0.179437000847,0.000172690172325,0.155271104219,0.0536610109272])
        w5 = torch.tensor([-0.369471756141,-0.124139165315,-1.63008285889,2.06268182076,1.05971997378,-0.404036285657,
                           0.35003581063,-0.269461692429,0.00170530533046,0.261731671162,0.108905459203])
    else:
        raise NotImplementedError
    return w1, w2, w3, w4, w5

def get_layer_1_neuron_biases(ver="2.1"):
    if ver =="2.1":
        b1 = torch.tensor(4.96238030555279)
        b2 = torch.tensor(1.416008443981500)
        b3 = torch.tensor(1.075897047213310)
        b4 = torch.tensor(1.533988264655420)
        b5 = torch.tensor(3.024115930757230)
    elif ver =="3A":
        b1 = torch.tensor(2.1161258270627883)
        b2 = torch.tensor(-1.2697541230231073)
        b3 = torch.tensor(-0.05084328062062615)
        b4 = torch.tensor(-1.3405631105862423)
        b5 = torch.tensor(0.00555729948301853)    
    elif ver =="3B":
        b1 = torch.tensor(-1.39740933966)
        b2 = torch.tensor(-0.728796420696)
        b3 = torch.tensor(-0.952390932098)
        b4 = torch.tensor(2.4023575151)
        b5 = torch.tensor(-0.0140976313556) 
    else:
        raise NotImplementedError
    return b1, b2, b3, b4, b5

def get_layer_2_bias(ver="2.1"):
    if ver =="2.1":
        bl2 = torch.tensor(1.096963107077220)
    elif ver =="3A":
        bl2 = torch.tensor(-0.2889370017570876)
    elif ver =="3B":
        bl2 = torch.tensor(0.497630893637)
    else:
        raise NotImplementedError
    return bl2

def get_layer_2_weights(ver="2.1"):
    if ver =="2.1":
        wl2 = torch.tensor([- 1.500135489728730, - 0.096283269121503, - 0.194935930577094, - 0.352305895755591, + 0.075107415847473,])
    elif ver =="3A":
        wl2 = torch.tensor([0.03766013804422397, -0.0006743151224540634, -0.000537335098594741, 0.6734767487882749, 0.06266448073894361])
    elif ver =="3B":
        wl2 = torch.tensor([0.495515257513,0.011460391566,-0.00615498084548,-0.923361561577,0.0904255753897])
    else:
        raise NotImplementedError
    return wl2

def get_norm_factors(ver="2.1"):
    if ver =="2.1":
        norm_factors = {"min_sample_B03" : 0, "max_sample_B03" : 0.253061520471542,
                        "min_sample_B04" : 0, "max_sample_B04" : 0.290393577911328,
                        "min_sample_B05" : 0, "max_sample_B05" : 0.305398915248555,
                        "min_sample_B06" : 0.006637972542253, "max_sample_B06" : 0.608900395797889,
                        "min_sample_B07" : 0.013972727018939, "max_sample_B07" : 0.753827384322927,
                        "min_sample_B8A" : 0.026690138082061, "max_sample_B8A" : 0.782011770669178,
                        "min_sample_B11" : 0.016388074192258, "max_sample_B11" : 0.493761397883092,
                        "min_sample_B12" : 0, "max_sample_B12" : 0.493025984460231,
                        "min_sample_viewZen" : 0.918595400582046, "max_sample_viewZen" : 1,
                        "min_sample_sunZen" : 0.342022871159208, "max_sample_sunZen" : 0.936206429175402,
                        "min_sample_lai" : 0.000319182538301, "max_sample_lai" : 14.4675094548151,}    
    elif ver =="3A":
        norm_factors = {"min_sample_B03" : 0, "max_sample_B03" : 0.23901527463861838,
                        "min_sample_B04" : 0, "max_sample_B04" : 0.29172736471507876,
                        "min_sample_B05" : 0, "max_sample_B05" : 0.32652671459255694,
                        "min_sample_B06" : 0.008717364330310326, "max_sample_B06" : 0.5938903910368211,
                        "min_sample_B07" : 0.019693160430621366, "max_sample_B07" : 0.7466909927207045,
                        "min_sample_B8A" : 0.026217828282102625, "max_sample_B8A" : 0.7582393779705984,
                        "min_sample_B11" : 0.018931934894415213, "max_sample_B11" : 0.4929337190581187,
                        "min_sample_B12" : 0, "max_sample_B12" : 0.4877499217101771,
                        "min_sample_viewZen" : 0.979624800125421, "max_sample_viewZen" : 1,
                        "min_sample_sunZen" : 0.342108564072183, "max_sample_sunZen" : 0.9274847491748729,
                        "min_sample_lai" : 0.00023377390882650673, "max_sample_lai" : 13.834592547008839,}
    elif ver =="3B":
        norm_factors = {"min_sample_B03" : 0, "max_sample_B03" : 0.247742161604,
                        "min_sample_B04" : 0, "max_sample_B04" : 0.305951681647,
                        "min_sample_B05" : 0, "max_sample_B05" : 0.327098829671,
                        "min_sample_B06" : 0.0119814116908, "max_sample_B06" : 0.599329840352,
                        "min_sample_B07" : 0.0169060342706, "max_sample_B07" : 0.741682769861,
                        "min_sample_B8A" : 0.0176448354545, "max_sample_B8A" : 0.780987637826,
                        "min_sample_B11" : 0.0147283842139, "max_sample_B11" : 0.507673379171,
                        "min_sample_B12" : 0, "max_sample_B12" : 0.502205128583,
                        "min_sample_viewZen" : 0.979624800125, "max_sample_viewZen" : 1,
                        "min_sample_sunZen" : 0.342108564072183, "max_sample_sunZen" : 0.9274847491748729,
                        "min_sample_lai" : 0.00023377390882650673, "max_sample_lai" : 13.834592547008839,}    
    else:
        raise NotImplementedError
    return norm_factors

def manage_extreme_values(lai, ver=2.1):
    if ver =="2.1":
        return lai
    elif ver =="3A" or ver == "3B":
        lai[torch.logical_and(lai < 0, lai >-0.2)] = 0
        lai[torch.logical_and(lai < 8.2, lai > 8)] = 8
        return lai
    else:
        raise NotImplementedError


def weiss_lai(s2_r, s2_a, band_dim=1, bands_idx=None, ver="2.1", lai_disp_norm=False):
    if bands_idx is None:
        B03 = s2_r.select(band_dim, 1).unsqueeze(band_dim)
        B04 = s2_r.select(band_dim, 2).unsqueeze(band_dim)
        B05 = s2_r.select(band_dim, 3).unsqueeze(band_dim)
        B06 = s2_r.select(band_dim, 4).unsqueeze(band_dim)
        B07 = s2_r.select(band_dim, 5).unsqueeze(band_dim)
        B8A = s2_r.select(band_dim, 7).unsqueeze(band_dim)
        B11 = s2_r.select(band_dim, 8).unsqueeze(band_dim)
        B12 = s2_r.select(band_dim, 9).unsqueeze(band_dim)
    else:
        B03 = s2_r.select(band_dim, bands_idx['B03']).unsqueeze(band_dim)
        B04 = s2_r.select(band_dim, bands_idx['B04']).unsqueeze(band_dim)
        B05 = s2_r.select(band_dim, bands_idx['B05']).unsqueeze(band_dim)
        B06 = s2_r.select(band_dim, bands_idx['B06']).unsqueeze(band_dim)
        B07 = s2_r.select(band_dim, bands_idx['B07']).unsqueeze(band_dim)
        B8A = s2_r.select(band_dim, bands_idx['B8A']).unsqueeze(band_dim)
        B11 = s2_r.select(band_dim, bands_idx['B11']).unsqueeze(band_dim)
        B12 = s2_r.select(band_dim, bands_idx['B12']).unsqueeze(band_dim)
    viewZenithMean = s2_a.select(band_dim, 1).unsqueeze(band_dim)
    sunZenithAngles = s2_a.select(band_dim, 0).unsqueeze(band_dim)
    relAzim = s2_a.select(band_dim, 2).unsqueeze(band_dim)
    
    norm_factors = get_norm_factors(ver=ver)
    w1, w2, w3, w4, w5 = get_layer_1_neuron_weights(ver=ver)
    b1, b2, b3, b4, b5 = get_layer_1_neuron_biases(ver=ver)
    wl2 = get_layer_2_weights(ver=ver)
    bl2 = get_layer_2_bias(ver=ver)

    b03_norm = normalize(B03, norm_factors["min_sample_B03"], norm_factors["max_sample_B03"])
    b04_norm = normalize(B04, norm_factors["min_sample_B04"], norm_factors["max_sample_B04"])
    b05_norm = normalize(B05, norm_factors["min_sample_B05"], norm_factors["max_sample_B05"])
    b06_norm = normalize(B06, norm_factors["min_sample_B06"], norm_factors["max_sample_B06"])
    b07_norm = normalize(B07, norm_factors["min_sample_B07"], norm_factors["max_sample_B07"])
    b8a_norm = normalize(B8A, norm_factors["min_sample_B8A"], norm_factors["max_sample_B8A"])
    b11_norm = normalize(B11, norm_factors["min_sample_B11"], norm_factors["max_sample_B11"])
    b12_norm = normalize(B12, norm_factors["min_sample_B12"], norm_factors["max_sample_B12"])
    viewZen_norm = normalize(torch.cos(torch.deg2rad(viewZenithMean)), norm_factors["min_sample_viewZen"], norm_factors["max_sample_viewZen"])
    sunZen_norm  = normalize(torch.cos(torch.deg2rad(sunZenithAngles)), norm_factors["min_sample_sunZen"], norm_factors["max_sample_sunZen"])
    relAzim_norm = torch.cos(torch.deg2rad(relAzim))

    x1 = torch.cat((b03_norm, b04_norm, b05_norm, b06_norm,b07_norm, b8a_norm, b11_norm, b12_norm,
                   viewZen_norm,sunZen_norm,relAzim_norm), axis=band_dim)
    nb_dim = len(b03_norm.size())
    n1 = neuron(x1, w1, b1, nb_dim, sum_dim=band_dim)
    n2 = neuron(x1, w2, b2, nb_dim, sum_dim=band_dim)
    n3 = neuron(x1, w3, b3, nb_dim, sum_dim=band_dim)
    n4 = neuron(x1, w4, b4, nb_dim, sum_dim=band_dim)
    n5 = neuron(x1, w5, b5, nb_dim, sum_dim=band_dim)

    l2 = layer2(n1, n2, n3, n4, n5, wl2, bl2, sum_dim=band_dim)
    lai = denormalize(l2, norm_factors["min_sample_lai"], norm_factors["max_sample_lai"])
    if lai_disp_norm:
        lai = lai / 3
    lai = manage_extreme_values(lai, ver=ver)
    return lai 

def neuron(x, weights, bias, nb_dim, sum_dim=1):
    weights = torch_select_unsqueeze(weights, sum_dim, nb_dim)
    sum =   bias + (weights * x).sum(sum_dim).unsqueeze(sum_dim)
    return tansig(sum)

def layer2(neuron1, neuron2, neuron3, neuron4, neuron5, weights, bias, sum_dim=1):
    x = torch.cat((neuron1,neuron2,neuron3,neuron4,neuron5), axis=sum_dim)
    weights = torch_select_unsqueeze(weights, sum_dim, len(neuron1.size()))
    sum = bias + (weights * x).sum(sum_dim).unsqueeze(sum_dim)
    return sum

def normalize(unnormalized, min_sample, max_sample): 
    return 2 * (unnormalized - min_sample) / (max_sample - min_sample) - 1

def denormalize(normalized, min_sample, max_sample): 
    return 0.5 * (normalized + 1) * (max_sample - min_sample) + min_sample

def tansig(input): 
    return 2 / (1 + torch.exp(-2 * input)) - 1 

from torchutils.patches import patchify, unpatchify
def get_weiss_lai(image_tensor, patch_size=32, bands=torch.tensor([0,1,2,3,4,5,6,7,8,9])):
    patched_tensor = patchify(image_tensor, patch_size=patch_size, margin=0)
    patched_lai_image = torch.zeros((patched_tensor.size(0), patched_tensor.size(1), 1, patch_size, patch_size)).to('cpu')
    for i in range(patched_tensor.size(0)):
        for j in range(patched_tensor.size(1)):
            x = patched_tensor[i,j, bands, :, :]
            angles = torch.zeros(3, patch_size, patch_size)
            angles[0,...] = patched_tensor[i, j, 11,...]
            angles[1,...] = patched_tensor[i, j, 13,...]
            angles[2,...] = patched_tensor[i, j, 12,...] - patched_tensor[i,j,14,...]
            with torch.no_grad():
                lai = weiss_lai(x, angles, band_dim=0)
            patched_lai_image[i,j,...] = lai
    lai_image = unpatchify(patched_lai_image)[:,:image_tensor.size(1),:image_tensor.size(2)]
    return lai_image

def compare_images():
    import matplotlib.pyplot as plt
    path_to_image = r"/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/data/validation_tiles/after_SENTINEL2B_20171127-105827-648_L2A_T31TCJ_C_V2-2_roi_0.pth"
    cropped_image = torch.load(path_to_image)[:,3:256-3,3:256-3]
    lai_image = get_weiss_lai(cropped_image, bands=torch.tensor([0,1,2,4,5,6,3,7,8,9]))

    import rasterio
    with rasterio.open('/home/yoel/Téléchargements/2017-11-27-00 00_2017-11-27-23 59_Sentinel-2_L2A_Custom_script.tiff', 'r') as ds:
        np_arr = ds.read()
        arr = torch.from_numpy(np_arr.astype(float))
    fig, ax = plot_patches((lai_image,arr),title_list=['NN lai', "Sentinel-Hub lai"], colorbar=True)
    fig.savefig('validation_weiss_nn_shub.png')

def main():
    import prosailvae
    import os
    import matplotlib.pyplot as plt
    from dataset.juan_datapoints import get_interpolated_validation_data
    juan_data_dir_path = os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/processed/"
    lai_min=0
    dt_max=10
    sites = ["france", "spain1", "italy1", "italy2"]
    device='cpu'

    weiss_mode=True
    list_lai_preds = []
    dt_list = []
    ndvi_list = []
    for site in sites:
        s2_r, s2_a, lais, dt = get_interpolated_validation_data(site, juan_data_dir_path, lai_min=lai_min, 
                                                                dt_max=dt_max, method="closest")
        b4 = s2_r[:,2]
        b8 = s2_r[:,6]
        ndvi = (b8-b4)/(b8+b4+1e-6)
        bands_idx = {'B02':0,
                     'B03':1,
                     'B04':2,
                     'B05':3,
                     'B06':4,
                     'B07':5,
                     'B08':6,
                     'B8A':7,
                     'B11':8,
                     'B12':9}
        if weiss_mode:
            s2_r = s2_r[:, torch.tensor([1,2,3,4,5,7,8,9])]
            bands_idx = {'B03':0,
                        'B04':1,
                        'B05':2,
                        'B06':3,
                        'B07':4,
                        'B8A':5,
                        'B11':6,
                        'B12':7}
        lai_pred = weiss_lai(s2_r, s2_a, bands_idx=bands_idx)
        list_lai_preds.append(torch.cat((lai_pred, lais), axis=1))
        dt_list.append(dt)
        ndvi_list.append(ndvi)
    list_lai_preds = torch.cat(list_lai_preds,axis=0)
    fig, ax = plt.subplots()
    from sklearn.metrics import r2_score
    import numpy as np
    fig, ax = plt.subplots()
    ax.scatter(list_lai_preds[:,1], 3*list_lai_preds[:,0],s=1)
    m, b = np.polyfit(list_lai_preds[:,1].numpy(), 3*list_lai_preds[:,0].numpy(), 1)
    r2 = r2_score(list_lai_preds[:,1].numpy(), 3*list_lai_preds[:,0].numpy())
    mse = (list_lai_preds[:,1] - 3*list_lai_preds[:,0]).pow(2).mean().numpy()
    ax.set_xlabel("LAI")
    ax.set_ylabel("LAI Predicted Sentinel-Hub")
    ax.plot([0,8],[0,8],'k--',)
    ax.plot([0,8],
            [m * 0 + b, m * 8 + b],'r', label="{:.2f} x + {:.2f}\n r2 = {:.2f}\n MSE: {:.2f}".format(m,b,r2,mse))
    ax.legend()
    ax.set_aspect('equal')
    fig.savefig('juan_shub_validation_altered.png')
    fig, ax = plt.subplots()
    ax.scatter(list_lai_preds[:,1], list_lai_preds[:,0],s=1)
    m, b = np.polyfit(list_lai_preds[:,1].numpy(), list_lai_preds[:,0].numpy(), 1)
    r2 = r2_score(list_lai_preds[:,1].numpy(), list_lai_preds[:,0].numpy())
    mse = (list_lai_preds[:,1] - list_lai_preds[:,0]).pow(2).mean().numpy()
    ax.set_xlabel("LAI")
    ax.set_ylabel("LAI Predicted Sentinel-Hub")
    ax.plot([0,8],[0,8],'k--',)
    ax.plot([0,8],
            [m * 0 + b, m * 8 + b],'r', label="{:.2f} x + {:.2f}\n r2 = {:.2f}\n MSE: {:.2f}".format(m,b,r2,mse))
    ax.legend()
    ax.set_aspect('equal')
    fig.savefig('juan_shub_validation.png')
    
    return

def weiss_regression():
    
    import os
    import prosailvae
    import numpy as np
    import pandas as pd
    def load_refl_angles(path_to_data_dir):
        path_to_file = path_to_data_dir + "/InputNoNoise_2.csv"
        assert os.path.isfile(path_to_file)
        df_validation_data = pd.read_csv(path_to_file, sep=" ", engine="python")
        n_obs = len(df_validation_data)
        s2_r = df_validation_data[['B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']].values
    
        tts = np.rad2deg(np.arccos(df_validation_data['cos(thetas)'].values))
        tto = np.rad2deg(np.arccos(df_validation_data['cos(thetav)'].values))
        psi = np.rad2deg(np.arccos(df_validation_data['cos(phiv-phis)'].values))
        lai = df_validation_data['lai_true'].values
        # lais = torch.as_tensor(df_validation_data['lai_true'].values.reshape(-1,1))
        # lai_bv_net = torch.as_tensor(df_validation_data['lai_bvnet'].values.reshape(-1,1))
        # time_delta = torch.zeros((n_obs,1))
        return s2_r, tts, tto, psi, lai
    def load_weiss_dataset(path_to_data_dir):
        s2_r, tts, tto, psi, lai = load_refl_angles(path_to_data_dir)
        s2_a = np.stack((tts,tto,psi),1)
        return s2_r, s2_a, lai
    s2_r, s2_a, lai = load_weiss_dataset(os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/")
    bands_idx = {'B03':0,
                        'B04':1,
                        'B05':2,
                        'B06':3,
                        'B07':4,
                        'B8A':5,
                        'B11':6,
                        'B12':7}
    lai_pred = weiss_lai(torch.from_numpy(s2_r), torch.from_numpy(s2_a), band_dim=1, bands_idx=bands_idx, ver="2.1", lai_disp_norm=False)
    import matplotlib.pyplot as plt 
    fig, ax = plt.subplots()
    ax.scatter(lai, lai_pred, s=0.5)
    ax.set_aspect('equal')
    ax.plot([0,14],[0,14],'k--')
    ax.set_xlabel('LAI Weiss Dataset')
    ax.set_ylabel('LAI Predicted by Sentinel Toolbox ver 2.1')
    fig.savefig(r"/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/shub_network_reg_weiss_dataset.png")
    return

def validate_sentinel_hub_lai():
    import rasterio
    import os
    import numpy as np
    raster_dir = r'/home/yoel/Téléchargements/'
    raster_files = ["b2_4.tiff", "b5_7.tiff","b8_11.tiff","b12_v.tiff","bs_lai.tiff"]
    arrs = []
    for filename in raster_files:
        path_to_raster = os.path.join(raster_dir, filename)
        with rasterio.open(path_to_raster, 'r') as ds:
            arr = ds.read()  # read all raster values
        arrs.append(arr)
    data = torch.from_numpy(np.concatenate(arrs,0))
    s2_r = data[:10,...]
    lai = data[-1,...]
    s2_a = torch.zeros((3, s2_r.shape[1], s2_r.shape[2]))
    s2_a[0,...] = data[12,...]
    s2_a[1,...] = data[10,...]
    s2_a[2,...] = (data[13,...] - data[11,...])
    lai_pred = weiss_lai(s2_r, s2_a, band_dim=0, bands_idx=None, ver="2.1", lai_disp_norm=True)
    
    import matplotlib.pyplot as plt
    from prosail_plots import plot_patches
    fig, ax = plot_patches((lai.unsqueeze(0), lai_pred.unsqueeze(0), lai.unsqueeze(0) - lai_pred.unsqueeze(0)))
    fig.savefig(r"/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/shub_lai_map.png")
    plt.figure()
    plt.imshow(s2_a[2,...])
    fig, ax = plt.subplots(10, figsize=(4,8), dpi = 150, sharex=True)
    for i in range(10):
        ax[i].scatter(s2_r[i,...].reshape(-1).cpu(), (lai - lai_pred).reshape(-1).cpu(), s=0.5)
    fig, ax = plt.subplots(10, figsize=(4,8), dpi = 150, sharex=True)
    for i in range(10):
        ax[i].hist(s2_r[i,...].reshape(-1).cpu(), bins=100)
    fig, ax = plt.subplots(3, figsize=(4,8), dpi = 150)
    for i in range(3):
        ax[i].scatter(s2_a[i,...].reshape(-1).cpu(), (lai - lai_pred).reshape(-1).cpu(), s=0.5)
    fig, ax = plt.subplots()
    ax.scatter(lai.reshape(-1)[(lai - lai_pred > 1e-3).reshape(-1)].cpu(), lai_pred.reshape(-1)[(lai - lai_pred > 1e-3).reshape(-1)].cpu(),s=0.5)
    ax.set_aspect('equal')
    fig, ax = plt.subplots()
    ax.scatter(lai, lai - lai_pred,s=0.5)
    fig, ax = plt.subplots()
    ax.scatter(lai_pred, lai - lai_pred,s=0.5)
    fig, ax = plt.subplots()
    ax.hist((lai - lai_pred).reshape(-1).cpu(),bins=100)
    return

def validate_snap_lai():
    import rasterio
    import os
    import numpy as np
    path_to_bands = r"/home/yoel/Documents/SNAP/results/subset_0_of_S2B_MSIL1C_20171127T105359_N0206_R051_T31TCJ_20171127T143619_resampled_6.tif"
    with rasterio.open(path_to_bands, 'r') as ds1:
        arr_bands = ds1.read()  # read all raster values
    s2_r = torch.from_numpy(arr_bands[:10,...])/10000
    s2_a = np.zeros((3, s2_r.shape[1], s2_r.shape[2]))
    s2_a[0,...] = arr_bands[12,...]
    s2_a[1,...] = arr_bands[10,...]
    s2_a[2,...] = (arr_bands[13,...] - arr_bands[11,...])
    s2_a = torch.from_numpy(s2_a)
    path_to_lai = r"/home/yoel/Documents/SNAP/results/subset_0_of_S2B_MSIL1C_20171127T105359_N0206_R051_T31TCJ_20171127T143619_resampled_biophysical.tif"
    with rasterio.open(path_to_lai, 'r') as ds:
        lai_arr = ds.read()  # read all raster values
    lai = torch.from_numpy(lai_arr[0,...]).unsqueeze(0)
    import matplotlib.pyplot as plt
    lai_pred = weiss_lai(s2_r, s2_a, band_dim=0, bands_idx=None, ver="3A", lai_disp_norm=False)
    from prosail_plots import plot_patches
    fig, ax = plot_patches((lai, lai_pred, lai - lai_pred), ["SNAP LAI", "S2tlbx python Prediction LAI (S2A)", "difference"])
    fig.savefig(r"/home/yoel/Documents/Dev/PROSAIL-VAE/prosailvae/results/snap_lai_map.png")
    fig, ax = plt.subplots()
    ax.scatter(lai.reshape(-1)[(lai - lai_pred > 1e-3).reshape(-1)].cpu(), lai_pred.reshape(-1)[(lai - lai_pred > 1e-3).reshape(-1)].cpu(),s=0.5)
    ax.set_aspect('equal')
    ax.set_xlabel("SNAP LAI")
    ax.plot([-1,10],[-1,10],'k--')
    ax.set_ylabel("S2tlbx python Prediction LAI (S2A)")

    fig, ax = plt.subplots()
    ax.scatter(lai, lai - lai_pred,s=0.5)
    ax.set_xlabel("SNAP LAI")
    ax.set_ylabel("Difference")
    return

if __name__ == "__main__":
    # validate_snap_lai()
    # weiss_regression()
    validate_sentinel_hub_lai()
    from prosail_plots import plot_patches
    compare_images()
    main()
