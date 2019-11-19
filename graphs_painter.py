import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np


num_planets = np.arange(1, 8)
populationSize = [10, 20, 30, 40, 50, 60, 100]
layers = [1, 2, 3, 4, 5, 10, 20]
importancePBest = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4]

error_popSize_linear = [0.001733241875555444, 0.0016156539538246853, 0.0015529498739469513, 0.0005903924021669087, 0.000775368739850973, 0.0005625358673777079, 0.0005248705777437337]
error_popSize_cubic = [0.016168864876614363, 0.017821279789111286, 0.008468003475707968, 0.008091299474214674, 0.009302160576079573, 0.011232180275837212, 0.0071422762249382]
error_popSize_sine = [0.008278968713477306, 0.0065796600988246206, 0.005394952831695273, 0.0051548989530690505, 0.00539352718609904, 0.005972330746381195, 0.0029384421044951186]
error_popSize_tanh = [6.783815320483055e-05, 4.808222780940105e-05, 4.338767864578588e-05, 4.592550820464327e-05, 1.1291559480332735e-05, 2.0655725864819966e-05, 1.5456005846661892e-05]
error_popSize_xor = [7.054204459501028e-06, 1.0073262581415063e-10, 6.264767863860223e-11, 1.3042492570609807e-13, 1.5141030203553887e-11, 9.850372716057206e-12, 9.132939268762731e-13]
error_popSize_complex = [0.05687611973163342, 0.0529669217707605, 0.050274082572706436, 0.043332384666356166, 0.044139096446994805, 0.046296158831459086, 0.04284244910122266]

error_layers_linear = [0.0006746080073655598, 0.0015256502264080485, 0.0020728451317561787, 0.0050718565848339395, 0.00934484535965646]
error_layers_cubic = [0.006764098163174741, 0.011285530997387525, 0.02303395891056537, 0.03460650808102679, 0.03601261265672832]
error_layers_sine = [0.0009330644332368552, 0.008957440198892691, 0.010726598218609783, 0.020903424213080134, 0.026222190743016034]
error_layers_tanh = [3.815888043881803e-06, 2.2146386522366296e-05, 0.0001364757703817887, 0.00032618037135514306, 0.0002664608597098348]
error_layers_xor = [1.2725050389408164e-21, 2.626470252550178e-13, 1.5746436032911958e-10, 1.99661221527249e-08, 1.719218111664287e-06]
error_layers_complex = [0.05179295524890428, 0.051624222658739415, 0.054510204764257386, 0.047479315226055016, 0.04352525947154507, 0.04043599883280295, 0.038569168030742555]

error_pbest_linear = [0.01429445292341397, 0.009675679055145457, 0.003509133803389635, 0.0043318470443657175, 0.0012888228401926772, 0.0013180089474503728, 0.0009454890501619853, 0.002060248942473276, 0.16298639204672455]
error_pbest_cubic = [0.025772438512515473, 0.01914257001743946, 0.02041278869739404, 0.0183975879188172, 0.01744957549652394, 0.006758197031916163, 0.016811518554193255, 0.015977045063304707, 0.0435424232606523]
error_pbest_sine = [0.02548704258408206, 0.031337104694914526, 0.020758373519260902, 0.01761400899218191, 0.010456845609703072, 0.007665262795851011, 0.0034330913636133054, 0.012156901436150384, 0.2923158081821879]
error_pbest_tanh = [0.00029624751896445265, 0.00030284124952602266, 0.00018695757853681906, 0.00011859041737342959, 0.0001519119026282377, 4.387549885907098e-05, 4.6936379518644165e-05, 4.642047673832335e-05, 0.5905923284616604]
error_pbest_xor = [2.3117170885160636e-17, 1.3908942728300614e-21, 9.666207751008651e-10, 6.329313929639658e-11, 1.9424645389847496e-14, 1.1026854484946324e-10, 1.6752440815008576e-12, 2.3432110800092207e-09, 0.16507032678544495]
error_pbest_complex = [0.05272807732417195, 0.055475534219294756, 0.052720672923824166, 0.05353696017424951, 0.05283811118365287, 0.05445465008946333, 0.047622937179473336, 0.047083150473980796, 0.045907271358459456]


fig, ((linear, cubic, sine), (tanh, xor, complexx)) = plt.subplots(2, 3)

# linear.plot(populationSize, error_popSize_linear, '-o', markersize=3, linewidth=1)
# cubic.plot(populationSize, error_popSize_cubic, '-o', markersize=3, linewidth=1)
# sine.plot(populationSize, error_popSize_sine, '-o', markersize=3, linewidth=1)
# tanh.plot(populationSize, error_popSize_tanh, '-o', markersize=3, linewidth=1)
# xor.plot(populationSize, error_popSize_xor, '-o', markersize=3, linewidth=1)
# complexx.plot(populationSize, error_popSize_complex, '-o', markersize=3, linewidth=1)

# linear.plot(layers[:5], error_layers_linear, '-o', markersize=3, linewidth=1, color='forestgreen')
# cubic.plot(layers[:5], error_layers_cubic, '-o', markersize=3, linewidth=1, color='forestgreen')
# sine.plot(layers[:5], error_layers_sine, '-o', markersize=3, linewidth=1, color='forestgreen')
# tanh.plot(layers[:5], error_layers_tanh, '-o', markersize=3, linewidth=1, color='forestgreen')
# xor.plot(layers[:5], error_layers_xor, '-o', markersize=3, linewidth=1, color='forestgreen')
# complexx.plot(layers, error_layers_complex, '-o', markersize=3, linewidth=1, color='forestgreen')

linear.plot(importancePBest, error_pbest_linear, '-o', markersize=3, linewidth=1, color='darkorange')
cubic.plot(importancePBest, error_pbest_cubic, '-o', markersize=3, linewidth=1, color='darkorange')
sine.plot(importancePBest, error_pbest_sine, '-o', markersize=3, linewidth=1, color='darkorange')
tanh.plot(importancePBest, error_pbest_tanh, '-o', markersize=3, linewidth=1, color='darkorange')
xor.plot(importancePBest, error_pbest_xor, '-o', markersize=3, linewidth=1, color='darkorange')
complexx.plot(importancePBest, error_pbest_complex, '-o', markersize=3, linewidth=1, color='darkorange')

linear.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
cubic.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
sine.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
tanh.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
xor.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))
complexx.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.e'))

linear.tick_params(axis='both', labelsize=7)
cubic.tick_params(axis='both', labelsize=7)
sine.tick_params(axis='both', labelsize=7)
tanh.tick_params(axis='both', labelsize=7)
xor.tick_params(axis='both', labelsize=7)
complexx.tick_params(axis='both', labelsize=7)

linear.set_title('Linear', fontsize=9)
cubic.set_title('Cubic', fontsize=9)
sine.set_title('Sine', fontsize=9)
tanh.set_title('Tanh', fontsize=9)
xor.set_title('XOR', fontsize=9)
complexx.set_title('Complex', fontsize=9)

linear.set_ylim(top=0.015, bottom=0)
cubic.set_ylim(top=0.028)
sine.set_ylim(top=0.035, bottom=0)
tanh.set_ylim(top=0.00035, bottom=0)
xor.set_ylim(top=2.5e-09, bottom=0)
complexx.set_ylim(top=0.056)

plt.show()
