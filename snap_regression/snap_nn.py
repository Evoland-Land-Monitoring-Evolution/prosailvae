from dataclasses import dataclass
import os
import torch
import torch.nn as nn
from collections import OrderedDict
from tqdm import trange
import numpy as np
import pandas as pd

def normalize(unnormalized:torch.Tensor, min_sample:torch.Tensor, max_sample:torch.Tensor):
    """
    Normalize with sample min and max of distribution
    """
    return 2 * (unnormalized - min_sample) / (max_sample - min_sample) - 1

def denormalize(normalized:torch.Tensor, min_sample:torch.Tensor, max_sample:torch.Tensor):
    """
    de-normalize with sample min and max of distribution
    """
    return 0.5 * (normalized + 1) * (max_sample - min_sample) + min_sample

@dataclass
class ParametersSnapLAINNV2:
    """
    Weights and biases of Snap NN's two layers
    """
    layer_1_weight: torch.Tensor = torch.tensor([[-0.0234068789665, 0.921655164636, 0.13557654408, -1.9383314724,
                                                  -3.34249581612, 0.90227764801, 0.205363538259, -0.0406078447217,
                                                  -0.0831964097271, 0.260029270774, 0.284761567219],
                                                 [-0.132555480857, -0.139574837334, -1.0146060169, -1.33089003865,
                                                  0.0317306245033, -1.43358354132, -0.959637898575, 1.13311570655,
                                                  0.216603876542, 0.410652303763, 0.0647601555435],
                                                 [0.0860159777249, 0.616648776881, 0.678003876447, 0.141102398645,
                                                  -0.0966822068835, -1.12883263886, 0.302189102741, 0.4344949373,
                                                  -0.0219036994906, -0.228492476802, -0.0394605375898],
                                                 [-0.10936659367, -0.0710462629727, 0.0645824114783, 2.90632523682,
                                                  -0.673873108979, -3.83805186828, 1.69597934453, 0.0469502960817,
                                                  -0.0497096526884, 0.021829545431, 0.0574838271041],
                                                 [-0.08993941616, 0.175395483106, -0.0818473291726, 2.21989536749,
                                                  1.71387397514, 0.7130691861, 0.138970813499, -0.060771761518,
                                                  0.124263341255, 0.210086140404, -0.1838781387]])
    
    layer_1_bias: torch.Tensor = torch.tensor([4.96238030555, 1.41600844398, 1.07589704721,
                                               1.53398826466, 3.02411593076])
    layer_2_weight: torch.Tensor = torch.tensor([[-1.50013548973,-0.0962832691215,-0.194935930577,-0.352305895756,0.0751074158475]])
    layer_2_bias: torch.Tensor = torch.tensor([1.09696310708])


@dataclass
class ParametersSnapLAINNV3B:
    """
    Weights and biases of Snap NN's two layers
    """
    layer_1_weight: torch.Tensor = torch.tensor([[0.0830079594419,-0.288281933525,-0.111764138832,-2.25405934742,
                                                  0.268135370597,3.14923063777,-1.06996677449,-0.1837991652,
                                                  0.0149841814287,0.0314033988537,0.00514853694825],
                                                 [1.00387064453,-0.813625826848,0.740886365416,1.00938214456,
                                                  -0.926530517091,-0.476882469139,-0.0798642487375,-1.01674056491,
                                                  -0.959418129249,0.749986744207,-1.2989372502],
                                                 [-0.914903852701,0.418794470829,0.156396260112,0.320562039933,
                                                  2.0695705545,1.6938367995,-0.307752980774,0.092487209186,
                                                  -2.07129204917,-1.00442058831,-0.531933380552],
                                                 [-0.000281553791956,0.250188709989,-0.151786460873,-0.8446482236,
                                                  -1.02084725541,0.00997687410719,-0.151504370026,0.179437000847,
                                                  0.000172690172325,0.155271104219,0.0536610109272],
                                                 [-0.369471756141,-0.124139165315,-1.63008285889,2.06268182076,
                                                  1.05971997378,-0.404036285657,0.35003581063,-0.269461692429,
                                                  0.00170530533046,0.261731671162,0.108905459203]])
    
    layer_1_bias: torch.Tensor = torch.tensor([-1.39740933966,-0.728796420696,-0.952390932098,
                                               2.4023575151,-0.0140976313556])
    layer_2_weight: torch.Tensor = torch.tensor([[0.495515257513,0.011460391566,-0.00615498084548,
                                                  -0.923361561577,0.0904255753897]])
    layer_2_bias: torch.Tensor = torch.tensor([0.497630893637])


@dataclass
class ParametersSnapLAINNV3A:
    """
    Weights and biases of Snap NN's two layers
    """
    layer_1_weight: torch.Tensor = torch.tensor([[-1.4710156196971582,-0.2563532709468933,-2.2180860576490495,
                                                  -1.085898980193225,1.476503732633888,-0.203130406532602,
                                                  1.9247696053465146,1.836433120470074,-0.4178313281505014,
                                                  0.11120379207368114,0.2710853575035239],
                                                 [1.8061903884637975,1.1335141059415899,0.7529844590307209,
                                                  -1.849059724727423,-2.3218651443747502,-0.14636054606555837,
                                                  -0.55740658813474,0.6841136954400282,0.8247349654342588,
                                                  -0.6080497708934546,0.32421327324338656],
                                                 [0.7970672934944534,0.4460131825143041,0.408586425658519,
                                                  -1.485087188758423,1.1623034379249386,-1.4153189332758485,
                                                  1.4595324130059253,0.3335744055337625,-1.6186285460736094,
                                                  -0.14549546441013367,-2.8578688750531547],
                                                 [0.09421670372484336,-0.22700197694341617,-0.15056963844888643,
                                                  -1.2842480628443311,0.6965044295491283,1.9782367938424654,
                                                  -0.8916551406435398,-0.1039817992235624,0.013484692816631437,
                                                  0.007841478637455265,-0.02581868621411949],
                                                 [-0.9667116308950019,-0.4183310913759375,-1.890580979987651,
                                                  0.0066940023538766685,1.9755057593307033,4.045480239195276,
                                                  -1.3979478080853602,-0.7644964732510922,0.010792479104631214,
                                                  1.1088706990036719,0.04211359644818833]])
    
    layer_1_bias: torch.Tensor = torch.tensor([2.1161258270627883,-1.2697541230231073,-0.05084328062062615,
                                               -1.3405631105862423,0.00555729948301853])
    layer_2_weight: torch.Tensor = torch.tensor([[0.03766013804422397,-0.0006743151224540634,-0.000537335098594741,
                                                  0.6734767487882749,0.06266448073894361]])
    layer_2_bias: torch.Tensor = torch.tensor([-0.2889370017570876])

@dataclass
class ParametersSnapCabNNV2:
    """
    Weights and biases of Snap NN's two layers for Cab
    """
    layer_1_weight: torch.Tensor = torch.tensor([[0.400396555257,0.607936279259,0.13746865078,-2.95586657346,
                                                  -3.18674668773,2.20680075125,-0.31378433614,0.256063547511,
                                                  -0.0716132198051,0.51011350421,0.142813982139],
                                                 [-0.250781102415,0.43908630292,-1.16059093752,-1.86193525027,
                                                  0.981359868452,1.63423083425,-0.872527934646,0.448240475035,
                                                  0.0370780835012,0.0300441896704,0.0059566866194],
                                                 [0.552080132569,-0.502919673167,6.10504192497,-1.29438611914,
                                                  -1.05995638835,-1.39409290242,0.324752732711,-1.75887182283,
                                                  -0.0366636798603,-0.183105291401,-0.0381453121174],
                                                 [0.211591184882,-0.248788896074,0.887151598039,1.14367589557,
                                                  -0.753968830338,-1.18545695308,0.541897860472,-0.252685834608,
                                                  -0.0234149010781,-0.0460225035496,-0.00657028408066],
                                                 [0.254790234231,-0.724968611431,0.731872806027,2.30345382102,
                                                  -0.849907966922,-6.42531550054,2.23884455846,-0.199937574298,
                                                  0.0973033317146,0.334528254938,0.113075306592]])

    layer_1_bias: torch.Tensor = torch.tensor([4.24229967016,-0.259569088226,3.13039262734,
                                               0.774423577182,2.58427664853])
    layer_2_weight: torch.Tensor = torch.tensor([[-0.352760040599,-0.603407399151,0.135099379384,
                                                  -1.73567312385,-0.147546813318]])
    layer_2_bias: torch.Tensor = torch.tensor([0.463426463934])


@dataclass
class ParametersSnapCabNNV3B:
    """
    Weights and biases of Snap NN's two layers for Cab
    """
    layer_1_weight: torch.Tensor = torch.tensor([[0.072663494226,7.70290146293,-8.9708798925,-7.12092525336,
                                                  -0.431432859098,-9.76621725119,-3.94343305737,5.4024983358,
                                                  0.317505127837,2.49642780027,-16.646596977],
                                                 [0.333355474719,0.831887523698,-4.9673822397,-3.31603614665,
                                                  6.03295530448,8.81808132523,0.826412940604,-1.62513574829,
                                                  0.124729764659,1.05717175678,-0.746056684848],
                                                 [0.151393270761,-0.238706388228,0.628860562857,2.37869751613,
                                                  -1.16507419767,-2.79368401251,0.950520196056,-0.257629879282,
                                                  -0.0156175301582,0.0285690311835,0.0265632084356],
                                                 [0.772385290557,-0.112838251687,2.86604615458,-0.368769877742,
                                                  -1.88470880737,-0.273445772555,0.359061409116,-0.177970813822,
                                                  0.0178560667205,-0.359202314618,0.0805162212937],
                                                 [-2.70886288775,6.39030158232,5.17544864537,-3.97773914527,
                                                  2.17235661127,4.30534465948,-5.64616550786,3.32370596624,
                                                  -5.65468679007,-14.6152796219,-10.2267142148]])

    layer_1_bias: torch.Tensor = torch.tensor([2.48093984595,3.49549708875,1.43146101049,
                                               1.51787023877,14.6720693643])
    layer_2_weight: torch.Tensor = torch.tensor([[-0.017072373108,0.0252014276913,-0.692000367377,
                                                  -0.130523852428,0.00199479745387]])
    layer_2_bias: torch.Tensor = torch.tensor([-0.14820880934])


@dataclass
class ParametersSnapCabNNV3A:
    """
    Weights and biases of Snap NN's two layers for Cab
    """
    layer_1_weight: torch.Tensor = torch.tensor([[17.96804419148567,-50.85215189472858,79.6258489045734,
                                                  93.90615783132985,-173.6207000430962,-133.31576631854682,
                                                  81.93480255666458,-60.07110409172156,19.29624317199184,
                                                  -15.563890950719589,-5.2152734207944835],
                                                 [57.736272159501716,8.74076507964802,72.76863546068279,
                                                  -5.595691263748469,-40.00955009275178,-40.775702150578994,
                                                  10.28827690495025,-22.702330737729234,-28.26799587664727,
                                                  -8.758181564687487,39.166871718330704],
                                                 [0.5699425125379961,0.8255188810088473,-0.8186606378663231,
                                                  -8.17977989706183,-1.2260880387853308,8.368384548920812,
                                                  -2.256846937359569,0.1425516990662458,0.03000302861289917,
                                                  -0.07146268866609334,-0.12757255375388069],
                                                 [3.2357945590327413,-6.374926310691868,18.822447100189333,
                                                  -5.928992135595644,-4.513160423336023,-23.84007074543908,
                                                  25.488266514059355,-8.042278344602638,19.968067450695848,
                                                  -33.33548374791842,12.224869437984701],
                                                 [0.1741658270177536,-0.19210084400794303,0.6805097278984471,
                                                  1.2260207866822954,-0.7956159603779578,-1.527327381554374,
                                                  0.6494803889605677,-0.20552191302882125,-0.0031181488054158866,
                                                  -0.016863548631218913,0.001411983943156535]])

    layer_1_bias: torch.Tensor = torch.tensor([-26.21580923479399,34.582038112515235,0.4717370390706491,
                                               -8.753297009060354,1.0818551080516499])
    layer_2_weight: torch.Tensor = torch.tensor([[0.021472311143093344,-0.0007822881987427077,-0.053883628973338815,
                                                  -0.0058358495064567435,-1.1566781736131353]])
    layer_2_bias: torch.Tensor = torch.tensor([0.08962056824850759])

@dataclass
class ParametersSnapCwNNV2:
    """
    Weights and biases of Snap NN's two layers for Cw
    """
    layer_1_weight: torch.Tensor = torch.tensor([[0.146378710426,1.18979928187,-0.906235139963,-0.808337508767,
                                                  -0.97333491783,-1.42591277646,-0.00561253629588,-0.634520356267,
                                                  -0.117226059989,-0.0602700912102,0.229407587132],
                                                 [0.283319173374,0.149342023041,1.08480588387,-0.138658791035,
                                                  -0.455759407329,0.420571438078,-1.7372949037,-0.704286287226,
                                                  0.0190953782358,-0.0393971316513,-0.00750241581744],
                                                 [-0.197487427943,-0.105460325978,0.158347670681,2.14912426654,
                                                  -0.970716842916,-4.92725317909,1.42034301781,1.45316917226,
                                                  0.0227257053609,0.269298650421,0.0849047657715],
                                                 [0.141405799763,0.33386260328,0.356218929123,-0.545942267639,
                                                  0.0891043076856,0.919298362929,-1.8520892625,-0.427539590779,
                                                  0.00791385646467,0.0148333201478,-0.00153786769736],
                                                 [-0.186781083395,-0.549163704901,-0.181287638772,0.96864043656,
                                                  -0.470442559117,-1.24859725244,2.67014942338,0.49009062438,
                                                  -0.00144931939526,0.00314829369692,0.0206517883893]])

    layer_1_bias: torch.Tensor = torch.tensor([-2.1064083686,-1.69022094794,3.10117655255,
                                               -1.31231626496,1.01131930348])
    layer_2_weight: torch.Tensor = torch.tensor([[-0.0775555890347,-0.86411786119,-0.199212415374,
                                                  1.98730461219,0.458926743489]])
    layer_2_bias: torch.Tensor = torch.tensor([-0.197591709977])

@dataclass
class ParametersSnapCwNNV3B:
    """
    Weights and biases of Snap NN's two layers for Cw
    """
    layer_1_weight: torch.Tensor = torch.tensor([[-0.345546365011,-0.351509810676,-0.148777714946,0.271352453066,
                                                  1.06515196712,-0.322743617992,0.636994542997,0.472342657829,
                                                  0.024431957672,-0.107106217115,-0.0234567664833],
                                                 [-0.0820653121734,0.329735143128,-0.0791352130978,0.55743449554,
                                                  1.29793378497,-0.724665412562,-1.46256292653,-0.0767521253483,
                                                  1.74656843221e-05,0.0557245540575,0.0154852401462],
                                                 [-0.0363073780514,-0.114820165172,0.831290806535,-2.85146192463,
                                                  0.433778039468,6.1668514761,-0.941681605083,-2.10127382183,
                                                  0.0433000837872,-0.264719581577,-0.0262179734111],
                                                 [-0.234122585859,-0.374765026698,-0.121696526606,1.76162889154,
                                                  -0.448040354415,-2.94327024371,3.90727595295,0.275495490926,
                                                  -0.0419265483036,0.131480780988,0.0255978298446],
                                                 [0.272870396301,0.430076329191,0.0183281712157,-0.621951608269,
                                                  -1.09595506664,1.39022034223,-0.887167093188,-0.432632371267,
                                                  -0.0268299468528,0.181911086692,0.0215595889021]])

    layer_1_bias: torch.Tensor = torch.tensor([1.79034000014,-1.24474337464,-3.28797061076,
                                               2.63369472353,-1.41733234784])
    layer_2_weight: torch.Tensor = torch.tensor([[1.2530768044,0.330688887223,0.175348509855,
                                                  -0.357923614781,0.926703868999]])
    layer_2_bias: torch.Tensor = torch.tensor([-0.43275732514])

@dataclass
class ParametersSnapCwNNV3A:
    """
    Weights and biases of Snap NN's two layers for Cw
    """
    layer_1_weight: torch.Tensor = torch.tensor([[0.7199263709034547,0.18455325143950238,1.3904459262777307,
                                                  -1.1650107373825371,0.12933341398732867,0.9303273837200489,
                                                  -2.197266308696758,-1.5353631648586703,0.03223275552109272,
                                                  -0.005165495624044329,-0.012192509221570474],
                                                 [-0.2612248274575229,0.04122793473345019,0.16898558754471674,
                                                  -0.3174253241532203,0.22275958929821343,1.8797192525197604,
                                                  -0.7831586987789828,0.3282733535681016,0.0019144400538951833,
                                                  0.12775416104222997,-0.0015026647746536695],
                                                 [0.13834330848083468,-0.36209458856789767,-0.23034570797564502,
                                                  2.466973326385496,-0.7063495644563834,-6.61548615890108,
                                                  2.078575603241158,1.909399185998267,0.028269818690235846,
                                                  0.2506183988870747,0.013205776513782276],
                                                 [-0.4554474190780372,-0.295762856489522,-0.6316680747872363,
                                                  1.0658107638909262,-0.25395955510564755,-1.3394536390630465,
                                                  2.431288494107135,0.8736704770694391,-0.016352310927536032,
                                                  -0.050840872724998064,0.003543294402900373],
                                                 [-0.13645352055632287,-0.4283173935839093,0.1194906805005869,
                                                  0.5509143587168239,-0.9097545554991361,-0.6550108111902776,
                                                  1.2541468041905215,1.0353713106516964,-0.22796341343136184,
                                                  0.9528764360833241,0.08741137848593626]])

    layer_1_bias: torch.Tensor = torch.tensor([-2.4512801805894013,0.9232050685742744,4.050238595137891,
                                               2.2212301929347062,3.104478923985122])
    layer_2_weight: torch.Tensor = torch.tensor([[-0.9061011452146178,0.1486555763338173,-0.15518493136962452,
                                                    -1.4761310858473666,-0.1724054832159909]])
    layer_2_bias: torch.Tensor = torch.tensor([-0.10003453640528981])


@dataclass
class NormSnapNNV2:
    """
    Min and max snap nn input and output for normalization
    """
    input_min: torch.Tensor = torch.tensor([0.0000,  0.0000,  0.0000,  0.00663797254225,
                                            0.0139727270189,  0.0266901380821,  0.0163880741923,  0.0000,
                                            0.918595400582,  0.342022871159, -1.0000])

    input_max: torch.Tensor = torch.tensor([0.253061520472,  0.290393577911,  0.305398915249,  0.608900395798,
                                            0.753827384323,  0.782011770669,  0.493761397883,  0.49302598446,
                                            1.0000,  0.936206429175,  1.0000])


class NormSnapNNV3B:
    """
    Min and max snap nn input and output for normalization
    """
    input_min: torch.Tensor = torch.tensor([0.0000,  0.0000,  0.0000,  0.0119814116908,
                                            0.0169060342706,  0.0176448354545,  0.0147283842139,  0.0000,
                                            0.979624800125,  0.342108564072, -1.0000])

    input_max: torch.Tensor = torch.tensor([0.247742161604,  0.305951681647,  0.327098829671,  0.599329840352,
                                            0.741682769861,  0.780987637826,  0.507673379171,  0.502205128583,
                                            1.0000,  0.927484749175,  1.0000])

class NormSnapNNV3A:
    """
    Min and max snap nn input and output for normalization
    """
    input_min: torch.Tensor = torch.tensor([0.0000,  0.0000,  0.0000,  0.008717364330310326,
                                            0.019693160430621366,  0.026217828282102625,  0.018931934894415213,  0.0000,
                                            0.979624800125421,  0.342108564072183, -1.0000])

    input_max: torch.Tensor = torch.tensor([0.23901527463861838,  0.29172736471507876,  0.32652671459255694,  0.5938903910368211,
                                            0.7466909927207045,  0.7582393779705984,  0.4929337190581187,  0.4877499217101771,
                                            1.0000,  0.9274847491748729,  1.0000])

@dataclass
class DenormSNAPLAIV2:
    lai_min: torch.Tensor = torch.tensor(0.000319182538301)
    lai_max: torch.Tensor = torch.tensor(14.4675094548)

@dataclass
class DenormSNAPLAIV3B:
    lai_min: torch.Tensor = torch.tensor(0.000233773908827)
    lai_max: torch.Tensor = torch.tensor(13.834592547)

@dataclass
class DenormSNAPLAIV3A:
    lai_min: torch.Tensor = torch.tensor(0.00023377390882650673)
    lai_max: torch.Tensor = torch.tensor(13.834592547008839)

@dataclass
class DenormSNAPCabV2:
    cab_min: torch.Tensor = torch.tensor(0.00742669295987) /10
    cab_max: torch.Tensor = torch.tensor(873.90822211) /10

@dataclass
class DenormSNAPCabV3B:
    cab_min: torch.Tensor = torch.tensor(0.0184770096032) /10
    cab_max: torch.Tensor = torch.tensor(888.156665152) /10

@dataclass
class DenormSNAPCabV3A:
    cab_min: torch.Tensor = torch.tensor(0.01847700960324858) /10
    cab_max: torch.Tensor = torch.tensor(888.1566651521919) /10

@dataclass
class DenormSNAPCwV2:
    cw_min: torch.Tensor = torch.tensor(3.85066859366e-06)
    cw_max: torch.Tensor = torch.tensor(0.522417054645)

@dataclass
class DenormSNAPCwV3B:
    cw_min: torch.Tensor = torch.tensor(2.84352788861e-06)
    cw_max: torch.Tensor = torch.tensor(0.419181347199)

@dataclass
class DenormSNAPCwV3A:
    cw_min: torch.Tensor = torch.tensor(4.227082600108468e-06)
    cw_max: torch.Tensor = torch.tensor(0.5229998511245837)

def get_SNAP_norm_factors(ver:str='2.1', variable='lai'):
    """
    Get normalization factor for SNAP NN
    """
    if ver == "2.1":
        snap_norm = NormSnapNNV2()
        if variable=="lai":
            variable_min = DenormSNAPLAIV2().lai_min
            variable_max = DenormSNAPLAIV2().lai_max
        elif variable=="cab":
            variable_min = DenormSNAPCabV2().cab_min
            variable_max = DenormSNAPCabV2().cab_max
        elif variable=="cw":
            variable_min = DenormSNAPCwV2().cw_min
            variable_max = DenormSNAPCwV2().cw_max
        else:
            raise NotImplementedError
    elif ver=="3B":
        snap_norm = NormSnapNNV3B()
        if variable=="lai":
            variable_min = DenormSNAPLAIV3B().lai_min
            variable_max = DenormSNAPLAIV3B().lai_max
        elif variable=="cab":
            variable_min = DenormSNAPCabV3B().cab_min
            variable_max = DenormSNAPCabV3B().cab_max
        elif variable=="cw":
            variable_min = DenormSNAPCwV3B().cw_min
            variable_max = DenormSNAPCwV3B().cw_max
        else:
            raise NotImplementedError
    elif ver=="3A":
        snap_norm = NormSnapNNV3A()
        if variable=="lai":
            variable_min = DenormSNAPLAIV3A().lai_min
            variable_max = DenormSNAPLAIV3A().lai_max
        elif variable=="cab":
            variable_min = DenormSNAPCabV3A().cab_min
            variable_max = DenormSNAPCabV3A().cab_max
        elif variable=="cw":
            variable_min = DenormSNAPCwV3A().cw_min
            variable_max = DenormSNAPCwV3A().cw_max
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    return snap_norm.input_min, snap_norm.input_max, variable_min, variable_max


class SnapNN(nn.Module):
    """
    Neural Network with SNAP architecture to predict LAI from S2 reflectances and angles
    """
    def __init__(self, device:str='cpu', ver:str="3A", variable='lai', second_layer=False):
        super().__init__()
        
        input_min, input_max, variable_min, variable_max = get_SNAP_norm_factors(ver=ver, variable=variable)
        input_size = len(input_max) # 8 bands + 3 angles
        hidden_layer_size = 5
        if not second_layer:
            layers = OrderedDict([
                    ('layer_1', nn.Linear(in_features=input_size, out_features=hidden_layer_size)),
                    ('tanh', nn.Tanh()),
                    ('layer_2', nn.Linear(in_features=hidden_layer_size, out_features=1))])
        else:
            layers = OrderedDict([
                    ('layer_0', nn.Linear(in_features=input_size, out_features=input_size)),
                    ('tanh', nn.Tanh()),
                    ('layer_1', nn.Linear(in_features=input_size, out_features=hidden_layer_size)),
                    ('tanh', nn.Tanh()),
                    ('layer_2', nn.Linear(in_features=hidden_layer_size, out_features=1))])
        self.input_min = input_min.to(device)
        self.input_max = input_max.to(device)
        self.variable_min = variable_min.to(device)
        self.variable_max = variable_max.to(device)
        self.net:nn.Sequential = nn.Sequential(layers).to(device)
        self.device = device
        self.ver = ver
        self.variable = variable

    def set_weiss_weights(self):
        """
        Set Neural Network weights and biases to SNAP's original values
        """
        if self.ver=="2.1":
            if self.variable=='lai':
                nn_parameters = ParametersSnapLAINNV2()
            elif self.variable=="cab":
                nn_parameters = ParametersSnapCabNNV2()
            elif self.variable=="cw":
                nn_parameters = ParametersSnapCwNNV2()
            else:
                raise NotImplementedError
        elif self.ver=="3A":
            if self.variable=='lai':
                nn_parameters = ParametersSnapLAINNV3A()
            elif self.variable=="cab":
                nn_parameters = ParametersSnapCabNNV3A()
            elif self.variable=="cw":
                nn_parameters = ParametersSnapCwNNV3A()
            else:
                raise NotImplementedError
        elif self.ver=="3B":
            if self.variable=='lai':
                nn_parameters = ParametersSnapLAINNV3B()
            elif self.variable=="cab":
                nn_parameters = ParametersSnapCabNNV3B()
            elif self.variable=="cw":
                nn_parameters = ParametersSnapCwNNV3B()
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        self.net.layer_1.bias = nn.Parameter(nn_parameters.layer_1_bias.to(self.device))
        self.net.layer_1.weight = nn.Parameter(nn_parameters.layer_1_weight.to(self.device))
        self.net.layer_2.bias = nn.Parameter(nn_parameters.layer_2_bias.to(self.device))
        self.net.layer_2.weight = nn.Parameter(nn_parameters.layer_2_weight.to(self.device))
    
    
    def forward(self, s2_data: torch.Tensor, spatial_mode=False):
        """
        Forward method of SNAP NN to predict a biophysical variable
        """
        if spatial_mode:
            if len(s2_data.size()) == 3:
                (_, size_h, size_w) = s2_data.size()
                s2_data = s2_data.permute(1,2,0).reshape(size_h * size_w,-1)
            else:
                raise NotImplementedError
        s2_data_norm = normalize(s2_data, self.input_min, self.input_max)
        variable_norm = self.net.forward(s2_data_norm)
        variable = denormalize(variable_norm, self.variable_min, self.variable_max)
        if spatial_mode:
            variable = variable.reshape(size_h, size_w, 1).permute(2,0,1)
        return variable

    def train_model(self, train_loader, valid_loader, optimizer, 
                    epochs:int=100, lr_scheduler=None, disable_tqdm:bool=False):
        """
        Fit and validate the model to data for a number of epochs
        """
        all_train_losses = []
        all_valid_losses = []
        all_lr = []
        for _ in trange(epochs, disable=disable_tqdm):
            train_loss = self.fit(train_loader, optimizer)
            all_train_losses.append(train_loss.item())
            valid_loss = self.validate(valid_loader)
            all_valid_losses.append(valid_loss.item())
            all_lr.append(optimizer.param_groups[0]['lr'])
            if lr_scheduler is not None:
                lr_scheduler.step(valid_loss)
            if all_lr[-1] <= 1e-8:
                break
        return all_train_losses, all_valid_losses, all_lr

    def fit(self, loader, optimizer):
        """
        Apply mini-batch optimization from a train dataloader
        """
        self.train()
        loss_mean = torch.tensor(0.0).to(self.device)
        for _, batch in enumerate(loader):
            loss = self.get_batch_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                loss_mean += loss / batch[0].size(0)
        return loss_mean

    def validate(self, loader):
        """
        Compute loss on loader with mini-batches
        """
        self.eval()
        with torch.no_grad():
            loss_mean = torch.tensor(0.0).to(self.device)
            for _, batch in enumerate(loader):
                loss = self.get_batch_loss(batch)
                loss_mean += loss / batch[0].size(0)
        return loss_mean

    def get_batch_loss(self, batch):
        """
        Computes loss on batch
        """
        s2_data, variable = batch
        variable_pred = self.forward(s2_data.to(self.device))
        return (variable_pred - variable.to(self.device)).pow(2).mean()


def load_refl_angles(path_to_data_dir: str):
    """
    Loads simulated s2 reflectance angles and LAI from weiss dataset.
    """
    path_to_file = path_to_data_dir + "/InputNoNoise_2.csv"
    assert os.path.isfile(path_to_file)
    df_validation_data = pd.read_csv(path_to_file, sep=" ", engine="python")
    s2_r = df_validation_data[['B3', 'B4', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']].values
    tts = np.rad2deg(np.arccos(df_validation_data['cos(thetas)'].values))
    tto = np.rad2deg(np.arccos(df_validation_data['cos(thetav)'].values))
    psi = np.rad2deg(np.arccos(df_validation_data['cos(phiv-phis)'].values))
    lai = df_validation_data['lai_true'].values
    return s2_r, tto, tts, psi, lai # Warning, inverted tto and tts w.r.t my prosil version

def load_weiss_dataset(path_to_data_dir: str):
    """
    Loads simulated s2 reflectance angles and LAI from weiss dataset as aggregated numpy arrays.
    """
    s2_r, tto, tts, psi, lai = load_refl_angles(path_to_data_dir)
    s2_a = np.stack((tto, tts, psi), 1)
    return s2_r, s2_a, lai

def test_snap_nn(ver="2.1"):
    """
    Test if SNAP neural network's outputs are identical to that of the translated java
    code of SNAP
    """
    from weiss_lai_sentinel_hub import (get_norm_factors, get_layer_1_neuron_weights, get_layer_1_neuron_biases,
                                        get_layer_2_weights, get_layer_2_bias, neuron, layer2) 
    import prosailvae
    s2_r, s2_a, lai = load_weiss_dataset(os.path.join(prosailvae.__path__[0], os.pardir) + "/field_data/lai/")
    snap_nn = SnapNN(ver=ver, variable="lai")
    snap_nn.set_weiss_weights()
    sample = torch.cat((torch.from_numpy(s2_r), torch.cos(torch.from_numpy(s2_a))), 1).float()
    ver=ver
    norm_factors = get_norm_factors(ver=ver)
    w1, w2, w3, w4, w5 = get_layer_1_neuron_weights(ver=ver)
    b1, b2, b3, b4, b5 = get_layer_1_neuron_biases(ver=ver)
    wl2 = get_layer_2_weights(ver=ver)
    bl2 = get_layer_2_bias(ver=ver)

    b03_norm = normalize(sample[:,0].unsqueeze(1), norm_factors["min_sample_B03"], norm_factors["max_sample_B03"])
    b04_norm = normalize(sample[:,1].unsqueeze(1), norm_factors["min_sample_B04"], norm_factors["max_sample_B04"])
    b05_norm = normalize(sample[:,2].unsqueeze(1), norm_factors["min_sample_B05"], norm_factors["max_sample_B05"])
    b06_norm = normalize(sample[:,3].unsqueeze(1), norm_factors["min_sample_B06"], norm_factors["max_sample_B06"])
    b07_norm = normalize(sample[:,4].unsqueeze(1), norm_factors["min_sample_B07"], norm_factors["max_sample_B07"])
    b8a_norm = normalize(sample[:,5].unsqueeze(1), norm_factors["min_sample_B8A"], norm_factors["max_sample_B8A"])
    b11_norm = normalize(sample[:,6].unsqueeze(1), norm_factors["min_sample_B11"], norm_factors["max_sample_B11"])
    b12_norm = normalize(sample[:,7].unsqueeze(1), norm_factors["min_sample_B12"], norm_factors["max_sample_B12"])
    viewZen_norm = normalize(sample[:,8].unsqueeze(1), norm_factors["min_sample_viewZen"], norm_factors["max_sample_viewZen"])
    sunZen_norm  = normalize(sample[:,9].unsqueeze(1), norm_factors["min_sample_sunZen"], norm_factors["max_sample_sunZen"])
    relAzim_norm = sample[:,10].unsqueeze(1)
    band_dim = 1
    with torch.no_grad():
        x_norm = normalize(sample, snap_nn.input_min, snap_nn.input_max)
        snap_input = torch.cat((b03_norm, b04_norm, b05_norm, b06_norm, b07_norm, b8a_norm, b11_norm, b12_norm,
                    viewZen_norm, sunZen_norm, relAzim_norm), axis=band_dim)
        assert torch.isclose(snap_input, x_norm, atol=1e-5,rtol=1e-5).all()
        nb_dim = len(b03_norm.size())
        neuron1 = neuron(snap_input, w1, b1, nb_dim, sum_dim=band_dim)
        neuron2 = neuron(snap_input, w2, b2, nb_dim, sum_dim=band_dim)
        neuron3 = neuron(snap_input, w3, b3, nb_dim, sum_dim=band_dim)
        neuron4 = neuron(snap_input, w4, b4, nb_dim, sum_dim=band_dim)
        neuron5 = neuron(snap_input, w5, b5, nb_dim, sum_dim=band_dim)
        linear_1_snap = nn.Linear(11,5)
        linear_1_snap.weight = snap_nn.net.layer_1.weight
        linear_1_snap.bias = snap_nn.net.layer_1.bias
        assert torch.isclose(linear_1_snap(x_norm), snap_nn.net.layer_1.bias
                             + x_norm @ snap_nn.net.layer_1.weight.transpose(1,0), atol=1e-4).all()

        n_snap_nn = torch.tanh(snap_nn.net.layer_1.bias + x_norm @ snap_nn.net.layer_1.weight.transpose(1,0))
        assert torch.isclose(n_snap_nn, torch.cat((neuron1, neuron2, neuron3, neuron4, neuron5), axis=1), atol=1e-4).all()

        linear_2_snap = snap_nn.net.layer_2
        # linear_2_snap.weight = snap_nn.net[2].weight
        # linear_2_snap.bias = snap_nn.net[2].bias
        assert torch.isclose(linear_2_snap(n_snap_nn), snap_nn.net.layer_2.bias
                             + n_snap_nn @ snap_nn.net.layer_2.weight.transpose(1,0), atol=1e-4).all()
        layer_2_output = layer2(neuron1, neuron2, neuron3, neuron4, neuron5, wl2, bl2, sum_dim=band_dim)
        l_snap_nn = snap_nn.net.layer_2.bias + n_snap_nn @ snap_nn.net.layer_2.weight.transpose(1,0)
        lai_prenorm_snap = snap_nn.net.forward(x_norm)
        assert torch.isclose(l_snap_nn.squeeze(), layer_2_output.squeeze(), atol=1e-4).all()
        assert torch.isclose(lai_prenorm_snap.squeeze(), layer_2_output.squeeze(), atol=1e-4).all()
        lai = denormalize(layer_2_output, norm_factors["min_sample_lai"], norm_factors["max_sample_lai"])
        snap_lai = denormalize(l_snap_nn, snap_nn.variable_min, snap_nn.variable_max)
        assert torch.isclose(snap_lai.squeeze(), lai.squeeze(), atol=1e-4).all()
        assert torch.isclose(snap_nn.forward(sample).squeeze(), lai.squeeze(), atol=1e-4).all()