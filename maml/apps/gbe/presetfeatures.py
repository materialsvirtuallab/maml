from sympy.physics.units.quantities import Quantity
import sympy.physics.units as u
from sympy.physics.units.systems import SI
from sympy import Symbol

# e_coh = FeatureUnit("e_coh", {'J': 1})
# a0 = FeatureUnit('a0', {'m': 1})
# ar = FeatureUnit('ar', {'m': 1})
# b0 = FeatureUnit('b0', {'m': 1})
# G = FeatureUnit('G', {'J': 1, 'm': -3})
# cos_theta = FeatureUnit('cos(theta)', {})
# sin_theta = FeatureUnit('sin(theta)', {})
# d_gb = FeatureUnit('d_gb', {'m': 1})
# d_rot = FeatureUnit('d_rot', {'m': 1})
# bbond_ratio = FeatureUnit('breakbond_ratio', {})
# mean_bl_chg = FeatureUnit('mean_bl_chg', {'m': 1})
# e_gb = FeatureUnit('gb_energy', {'J': 1, 'm': -2})

# e_coh = Quantity(Symbol("e_coh"), latex_repr='E_{coh}')
# a0 = Quantity(Symbol('a_0'), latex_repr='a_0')
# ar = Quantity(Symbol('a_r'), latex_repr='ar')
# b0 = Quantity(Symbol('b_0'), latex_repr='b_0')
# G = Quantity(Symbol('G'), latex_repr='G')
# cos_theta = Quantity(Symbol('cos(theta)'), latex_repr='cos(theta)')
# sin_theta = Quantity(Symbol('sin(theta)'), latex_repr='sin(theta)')
# d_gb = Quantity(Symbol('d_gb'), latex_rep='d_{gb}')
# d_rot = Quantity(Symbol('d_rot'), latex_rep='d_{rot}')
# bbond_ratio = Quantity(Symbol('bbond_ratio'), latex_rep='bbond_ratio')
# mean_delta_bl = Quantity(Symbol('mean_delta_bl'), latex_rep='mean_delta_bl')
# e_gb = Quantity(Symbol("e_gb"), latex_repr='E_{gb}')
# norm_e_gb = Quantity(Symbol("e_gb/e_coh", latex_repr="E_{gb}/E_{coh}"))
# rnorm_e_gb = Quantity(Symbol("e_coh/e_gb", latex_repr="E_{coh}/E_{gb}"))

# Quantities = [e_coh, a0, ar, b0, G, cos_theta, sin_theta,
#               d_gb, d_rot, bbond_ratio, mean_delta_bl, hb, bdensity,
#               CLTE]
#
# for q in [a0, ar, b0, d_gb, d_rot, mean_delta_bl]:
#     SI.set_quantity_dimension(q, u.length)
#     SI.set_quantity_scale_factor(q, u.nm / 10)
#
# for q in [cos_theta, sin_theta, bbond_ratio]:
#     SI.set_quantity_dimension(q, 1)
#     SI.set_quantity_scale_factor(q, 1)
#
# SI.set_quantity_dimension(hb, u.force / (u.length ** 2))
# SI.set_quantity_scale_factor(hb, u.newton / (u.m ** 2))
#
# SI.set_quantity_dimension(bdensity, u.mass / u.volume)
# SI.set_quantity_scale_factor(bdensity, u.gram / (u.centimeter ** 3) )
#
# SI.set_quantity_dimension(CLTE, 1 / u.temperature)
# SI.set_quantity_scale_factor(CLTE, 1 / u.kelvin)
#
# SI.set_quantity_dimension(e_coh, u.energy)
# SI.set_quantity_scale_factor(e_coh, u.joule)
#
# SI.set_quantity_dimension(G, u.energy / (u.length ** 3))
# SI.set_quantity_scale_factor(G, u.joule / (u.m ** 3))
#
# SI.set_quantity_dimension(e_gb, u.energy / (u.length ** 2))
# SI.set_quantity_scale_factor(e_gb, u.joule / (u.m ** 2))
#
# SI.set_quantity_dimension(norm_e_gb, 1 / (u.length ** 2))
# SI.set_quantity_scale_factor(norm_e_gb, 1/ (u.m ** 2))
#
# SI.set_quantity_dimension(rnorm_e_gb, (u.length ** 2))
# SI.set_quantity_scale_factor(rnorm_e_gb, (u.m ** 2))
#
# Qdim_dict = {q: q.dimension for q in Quantities}

class my_quant:
    def __init__(self, str_name: str, latex_name: str=None, latex_unit: str=None):
        self.str_name = str_name
        self.latex_name = latex_name
        self.latex_unit = latex_unit

    @property
    def name(self):
        return self.str_name
    @property
    def latex(self):
        return self.latex_name
    @property
    def unit(self):
        return self.latex_unit


initial_fnames = [r'$d_{GB}$',
                  r'$d_{rot}$',
                 r'cos($\theta$)',
                  r'$E_{coh}$',
                  r'$\widebar{{\Delta} (BL)}$',
                  r'$G$',
                  r'$a_{0}$',
                  r'$\gamma_{GB}^{DFT}$',]
str2lat = {'d_gb': r'$d_{GB}$',
           'd_rot': r'$d_{rot}$',
           'sin(theta)': r'sin($\theta$)',
           'cos(theta)': r'cos($\theta$)',
           'e_coh': r'$E_{coh}$',
           'G': r'$G$',
           'a_0': r'$a_{0}$',
           'a_r': r'$a_{r}$',
           'mean_delta_bl': r'$\bar{{\Delta} (BL)}$',
           'mean_bl':r'$\bar{BL}$',
          'Density': 'Density',
           'e_gb': '$\gamma_{GB}$'}
features = ['d_gb', 'd_rot', 'cos(theta)', 'mean_delta_bl',
            'G', 'mean_bl']

e_coh = my_quant("e_coh", str2lat['e_coh'], "J")
a0 = my_quant("a_0", str2lat['a_0'], "$\AA$")
ar = my_quant("a_r", str2lat['a_r'], "$\AA$")
cos_theta = my_quant("cos(theta)", str2lat['cos(theta)'])
sin_theta = my_quant("sin(theta)", str2lat['sin(theta)'])
d_gb = my_quant("d_gb", str2lat["d_gb"])
d_rot = my_quant("d_rot", str2lat["d_rot"])
mean_delta_bl = my_quant("mean_delta_bl", str2lat["mean_delta_bl"], "$\AA$")
G = my_quant("G", str2lat['G'], "J m$^{-3}$")
e_gb = my_quant("e_gb", "$E_{GB}$", "J m$^{-2}$")
hb = my_quant("HB", "HB", "")
bdensity = my_quant("Density", "$Density$", "g cm$^{-3}$")
CLTE = my_quant("CLTE", "$CLTE", "")
mean_bl = my_quant("mean_bl", str2lat["mean_bl"], "$\AA$")
