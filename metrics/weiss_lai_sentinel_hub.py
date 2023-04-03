from math import pi 
import torch
from prosailvae.utils import torch_select_unsqueeze
    
def weiss_lai(s2_r, s2_a, band_dim = 1):
    B03 = s2_r.select(band_dim, 1).unsqueeze(band_dim)
    B04 = s2_r.select(band_dim, 2).unsqueeze(band_dim)
    B05 = s2_r.select(band_dim, 3).unsqueeze(band_dim)
    B06 = s2_r.select(band_dim, 4).unsqueeze(band_dim)
    B07 = s2_r.select(band_dim, 5).unsqueeze(band_dim)
    B8A = s2_r.select(band_dim, 7).unsqueeze(band_dim)
    B11 = s2_r.select(band_dim, 8).unsqueeze(band_dim)
    B12 = s2_r.select(band_dim, 9).unsqueeze(band_dim)

    viewZenithMean = s2_a.select(band_dim, 1).unsqueeze(band_dim)
    sunZenithAngles = s2_a.select(band_dim, 0).unsqueeze(band_dim)
    relAzim = s2_a.select(band_dim, 2).unsqueeze(band_dim)
    b03_norm = normalize(B03, 0, 0.253061520471542)
    b04_norm = normalize(B04, 0, 0.290393577911328)
    b05_norm = normalize(B05, 0, 0.305398915248555)
    b06_norm = normalize(B06, 0.006637972542253, 0.608900395797889)
    b07_norm = normalize(B07, 0.013972727018939, 0.753827384322927)
    b8a_norm = normalize(B8A, 0.026690138082061, 0.782011770669178)
    b11_norm = normalize(B11, 0.016388074192258, 0.493761397883092)
    b12_norm = normalize(B12, 0, 0.493025984460231)
    viewZen_norm = normalize(torch.cos(torch.deg2rad(viewZenithMean)), 0.918595400582046, 1)
    sunZen_norm  = normalize(torch.cos(torch.deg2rad(sunZenithAngles)), 0.342022871159208, 0.936206429175402)
    relAzim_norm = torch.cos(relAzim)

    n1 = neuron1(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=band_dim)
    n2 = neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=band_dim)
    n3 = neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=band_dim)
    n4 = neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=band_dim)
    n5 = neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=band_dim)

    l2 = layer2(n1, n2, n3, n4, n5, sum_dim=band_dim)

    lai = denormalize(l2, 0.000319182538301, 14.4675094548151)
    return lai / 3


def neuron1(b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm, 
            viewZen_norm, sunZen_norm, relAzim_norm, sum_dim=1):
    weights = torch.tensor([- 0.023406878966470, 
                            + 0.921655164636366,
                            + 0.135576544080099, 
                            - 1.938331472397950, 
                            - 3.342495816122680,
                            + 0.902277648009576,
                            + 0.205363538258614,
                            + 0.040607844721716,
                            + 0.083196409727092,
                            + 0.260029270773809,
                            + 0.284761567218845])
    weights = torch_select_unsqueeze(weights, sum_dim, len(b03_norm.size()))
    bias = torch.tensor(4.96238030555279)
    x = torch.cat((b03_norm, b04_norm, b05_norm, b06_norm,b07_norm, b8a_norm, b11_norm, b12_norm,
                   viewZen_norm,sunZen_norm,relAzim_norm), axis=sum_dim)
    sum =   bias + (weights * x).sum(sum_dim).unsqueeze(sum_dim)

    return tansig(sum)


def neuron2(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=1):
    bias = torch.tensor(1.416008443981500)
    weights = torch.tensor([- 0.132555480856684,
                            - 0.139574837333540,
                            - 1.014606016898920,
                            - 1.330890038649270,
                            + 0.031730624503341,
                            - 1.433583541317050,
                            - 0.959637898574699,
                            + 1.133115706551000,
                            + 0.216603876541632,
                            + 0.410652303762839,
                            + 0.064760155543506])
    
    x = torch.cat((b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,
                   b11_norm,b12_norm,viewZen_norm,sunZen_norm,relAzim_norm), axis=sum_dim)
    weights = torch_select_unsqueeze(weights, sum_dim, len(b03_norm.size()))
    sum = bias + (weights * x).sum(sum_dim).unsqueeze(sum_dim)
    

    return tansig(sum)


def neuron3(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=1): 
    bias = torch.tensor(1.075897047213310)
    weights = torch.tensor([  0.086015977724868,
                            + 0.616648776881434,
                            + 0.678003876446556,
                            + 0.141102398644968,
                            - 0.096682206883546,
                            - 1.128832638862200,
                            + 0.302189102741375,
                            + 0.434494937299725,
                            - 0.021903699490589,
                            - 0.228492476802263,
                            - 0.039460537589826,]) 
    x = torch.cat((b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,
                   b11_norm,b12_norm,viewZen_norm,sunZen_norm,relAzim_norm), axis=sum_dim)
    weights = torch_select_unsqueeze(weights, sum_dim, len(b03_norm.size()))
    sum = bias + (weights * x).sum(sum_dim).unsqueeze(sum_dim)
    return tansig(sum)


def neuron4(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=1):
    bias = torch.tensor(1.533988264655420)
    weights = torch.tensor([- 0.109366593670404,
                            - 0.071046262972729,
                            + 0.064582411478320,
                            + 2.906325236823160,
                            - 0.673873108979163,
                            - 3.838051868280840,
                            + 1.695979344531530,
                            + 0.046950296081713,
                            - 0.049709652688365,
                            + 0.021829545430994,
                            + 0.057483827104091])
    x = torch.cat((b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,
                   b11_norm,b12_norm,viewZen_norm,sunZen_norm,relAzim_norm), axis=sum_dim)
    weights = torch_select_unsqueeze(weights, sum_dim, len(b03_norm.size()))
    sum = bias + (weights * x).sum(sum_dim).unsqueeze(sum_dim)
    return tansig(sum)


def neuron5(b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,b12_norm, viewZen_norm,sunZen_norm,relAzim_norm, sum_dim=1): 
    bias = torch.tensor(3.024115930757230)
    weights = torch.tensor([- 0.089939416159969,
                            + 0.175395483106147,
                            - 0.081847329172620,
                            + 2.219895367487790,
                            + 1.713873975136850,
                            + 0.713069186099534,
                            + 0.138970813499201,
                            - 0.060771761518025,
                            + 0.124263341255473,
                            + 0.210086140404351,
                            - 0.183878138700341,])
    x = torch.cat((b03_norm,b04_norm,b05_norm,b06_norm,b07_norm,b8a_norm,b11_norm,
                   b12_norm,viewZen_norm,sunZen_norm,relAzim_norm), axis=sum_dim)
    weights = torch_select_unsqueeze(weights, sum_dim, len(b03_norm.size()))
    sum = bias + (weights * x).sum(sum_dim).unsqueeze(sum_dim)
    return tansig(sum)


def layer2(neuron1, neuron2, neuron3, neuron4, neuron5, sum_dim=1):
    bias = torch.tensor(1.096963107077220)
    weights = torch.tensor([- 1.500135489728730,
                            - 0.096283269121503,
                            - 0.194935930577094,
                            - 0.352305895755591,
                            + 0.075107415847473,])
    x = torch.cat((neuron1,neuron2,neuron3,neuron4,neuron5))
    weights = torch_select_unsqueeze(weights, sum_dim, len(neuron1.size()))
    print(weights.size())
    print(x.size())
    sum = bias + (weights * x).sum(sum_dim).unsqueeze(sum_dim)
    return sum


def normalize(unnormalized, min_sample, max_sample): 
    return 2 * (unnormalized - min_sample) / (max_sample - min_sample) - 1

def denormalize(normalized, min_sample, max_sample): 
    return 0.5 * (normalized + 1) * (max_sample - min_sample) + min_sample

def tansig(input): 
    return 2 / (1 + torch.exp(-2 * input)) - 1 

