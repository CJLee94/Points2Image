import math, random
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from tqdm import tqdm
from skimage.filters import gaussian
from scipy.stats import beta
from bayes_opt import BayesianOptimization
from scipy.ndimage import distance_transform_cdt


class OVERLAP_AREA:
    def __init__(self, mask):
        self.mask=mask
        self.distance_map = self.get_distance_map()
        self.state_dict = None
        self.label = 1
        
    def set_param(self, **kwargs):
        self.state_dict = kwargs
        self.mask[self.state_dict['x'], self.state_dict['y']] = 0
        self.distance_map = self.get_distance_map()
        
    def __call__(self, ecc, angle):
        ellps = draw_polygon(ecc=ecc, angle=angle, s=self.mask.shape[0], **self.state_dict)
#         ellps = draw_ellipse(center=self.center,
#                              a=a,
#                              b=b,
#                              angle=angle,
#                              mask_shape=self.mask.shape,
#                             )
        ellps = ellps.astype(bool)
        mask = self.mask>0
        return -(ellps&mask).sum()/(self.distance_map[ellps].sum()/10+1e-6)
    
    def draw(self, ecc, angle):
        ellps = draw_polygon(ecc=ecc, angle=angle, s=self.mask.shape[0], **self.state_dict)
        self.mask[ellps.astype(bool)] = self.label
        self.distance_map = self.get_distance_map()
        self.label += 1
    
    def get_mask(self):
        return self.mask
    
    def get_distance_map(self):
        return distance_transform_cdt(~(self.mask>0))

    
def random_polygon_params():
    area, irr, spike, nverti = get_rand_polygon_param(nsize, **kwargs);

def generate_fake_mask_from_points(points, nsize, var_x=0, var_y=0, ntry_max=30, **kwargs):
    bin_mask = np.zeros((nsize,nsize))
    color_mask = np.zeros((nsize,nsize))

    density = np.zeros((nsize, nsize), dtype=np.uint8)
    for c_row, c_col in points:
        density[int(c_row), int(c_col)] = 255
    ov = OVERLAP_AREA(density)
    density = gaussian(density, 20)
    
    with tqdm(total=len(points)) as pbar:
        for nuclei_no, (x, y) in enumerate(points):
            mu_Area = 1000*np.exp(-1000*np.array(density[x, y]))+100
            var_Area = 1700000*np.exp(-2800*np.array(density[x, y]))+100
            max_Area = 8000*np.exp(-1300*np.array(density[x, y]))+200
            area, irr, spike, nverti = get_rand_polygon_param(mu_Area=mu_Area, sigma_Area=var_Area, max_Area = max_Area,)
            
            ov.set_param(x=x, y=y, area=area, irr=irr, spike=spike, nverti=nverti)
#             ecc = np.random.rand()
#             angle = random.uniform(0, 2*math.pi)
            
            pbounds = {'ecc':[0.5,1], 'angle':[0,180]}
            optimizer = BayesianOptimization(
                f=ov,
                pbounds=pbounds,
                verbose=0,
                random_state=nuclei_no,
            )
            optimizer.maximize(
                init_points=1,
                n_iter=ntry_max,
            )
            ov.draw(**optimizer.max['params'])
#             ov.draw(ecc, angle)
            
            
            
#             overlap = np.sum((im_add*bin_mask)>0) / (np.sum(im_add>0).astype(np.float32))
#             ntry = 0
#             while (overlap>0.05) and (ntry<ntry_max):
#                 im_add = rand_nucleus(x=x, y=y, nsize = nsize, 
#                                       mu_Area=mu_Area, 
#                                       sigma_Area=var_Area, 
#                                       max_Area = max_Area,
#                                       **kwargs)
#                 overlap = np.sum((im_add*bin_mask)>0) / (np.sum(im_add>0).astype(np.float32))
#                 ntry += 1
#             color_mask[im_add.astype(bool)] = nuclei_no+1
#             bin_mask[im_add.astype(bool)] = 1
# #             contour += con_add
            pbar.update(1)
    return ov


def generate_fake_mask_from_points_old(points, nsize, ntry_max = 50, **kwargs):
    bin_mask = np.zeros((nsize,nsize,3))
    color_mask = np.zeros((nsize,nsize,3))
    contour = np.zeros((nsize,nsize,3))
    with tqdm(total=len(points)) as pbar:
        for nuclei_no, (x, y) in enumerate(points):
            im_add, con_add = rand_nucleus(x=x, y=y, nsize = nsize, **kwargs)
            overlap = np.sum((im_add*bin_mask)>0) / (np.sum(im_add>0).astype(np.float32))
            ntry = 0
            while (overlap>0.05) and (ntry<ntry_max):
                im_add, con_add = rand_nucleus(x=x, y=y, nsize=nsize, **kwargs)
                overlap = np.sum((im_add*bin_mask)>0) / (np.sum(im_add>0).astype(np.float32))
                ntry += 1
            color_mask += im_add * np.random.rand(3)[None,None]
            bin_mask += im_add
            contour += con_add
            pbar.update(1)
    return color_mask, bin_mask, contour


def get_rand_polygon_param(mu_Area=600, sigma_Area=400, max_Area = 500,
                           mu_irr=0.6, sigma_irr=2.0, 
                           mu_spike=0.04, sigma_spike=0.005, 
                           mu_nverti=20, sigma_nverti=8):
    beta_mean = mu_Area/max_Area
    beta_var = sigma_Area/max_Area**2
    beta_a = beta_mean**2*(1-beta_mean)/beta_var-beta_mean
    beta_b = beta_mean*(1-beta_mean)**2/beta_var-1+beta_mean
    if beta_a < 0 or beta_b < 0:
        import pdb
        pdb.set_trace()
    area = beta.rvs(beta_a, beta_b)*max_Area;
    irr = mu_irr+random.random()*sigma_irr;
    spike = mu_spike+random.random()*sigma_spike;
    nverti = int(random.random()*sigma_nverti+mu_nverti);
    return area, irr, spike, nverti;


def draw_polygon(x=None, y=None, area=None, ecc=None, 
                 irr=None, spike=None, nverti=None, s=None, angle=None):
    angle = angle/180.0*math.pi
    vertices = generatePolygon(x, y, area, ecc, irr, spike, nverti, angle);
    mask = Image.fromarray(np.zeros((s, s), dtype=np.uint8));
    draw = ImageDraw.Draw(mask);
    draw.polygon(vertices, fill=1);
    return np.array(mask);

def rand_nucleus(x, y, ecc=0, angle=0, nsize = 460, **kwargs):
    area, irr, spike, nverti = get_rand_polygon_param(nsize, **kwargs);
    mask = draw_polygon(x, y, area, ecc, irr, spike, nverti, nsize, angle);
    return mask


def generatePolygon(ctrX, ctrY, Area, eccentricity, irregularity, spikeyness, numVerts, angle):
    irregularity = clip( irregularity, 0,1 ) * 2*math.pi / numVerts
    eccentricity = clip(eccentricity, 0, 1)
    semi_minor = math.sqrt(Area*math.sqrt(1-eccentricity**2)/math.pi)

    # generate n angle steps
    angleSteps = []
    lower = (2*math.pi / numVerts) - irregularity
    upper = (2*math.pi / numVerts) + irregularity
    angle_sum = 0
    for i in range(numVerts):
        tmp = random.uniform(lower, upper)
        angleSteps.append( tmp )
        angle_sum = angle_sum + tmp
    
    # normalize the steps so that point 0 and point n+1 are the same
    k = angle_sum / (2*math.pi)
    for i in range(numVerts):
        angleSteps[i] = angleSteps[i] / k
        
    points = []
#     angle = random.uniform(0, 2*math.pi)
    init_angle = angle

    for i in range(numVerts):
        r_i = semi_minor//(math.sqrt(1-(eccentricity*math.cos(angle))**2)+1e-6)
        tmp_spikeyness = clip(spikeyness, 0, 1) * r_i
        r_i = clip(random.gauss(r_i, tmp_spikeyness), 0, 2*semi_minor)
        x = ctrX + r_i*math.cos(angle-init_angle)
        y = ctrY + r_i*math.sin(angle-init_angle)
        points.append((int(x),  int(y)))

        angle = angle + angleSteps[i]
    
    return points
    


def clip(x, min, max):
    if( min > max ):  return x
    elif( x < min ):  return min
    elif( x > max ):  return max
    else:             return x
