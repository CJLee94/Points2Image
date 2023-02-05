import math, random
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from tqdm import tqdm
from skimage.filters import gaussian
from scipy.stats import beta

def generate_fake_mask_from_points(points, nsize, density=None, ntry_max = 50, **kwargs):
    bin_mask = np.zeros((nsize,nsize,3))
    color_mask = np.zeros((nsize,nsize,3))
    contour = np.zeros((nsize,nsize,3))

    if density is None:
        density = np.zeros((nsize, nsize), dtype=np.uint8)
        for c_row, c_col in points:
            density[int(c_row), int(c_col)] = 255
        density = gaussian(density, 20)
    with tqdm(total=len(points)) as pbar:
        for nuclei_no, (x, y) in enumerate(points):
            # size_scale = np.exp(-1200*density[x, y])
            mu_Area = 1000*np.exp(-1000*np.array(density[x, y]))+100
            var_Area = 1700000*np.exp(-2800*np.array(density[x, y]))+100
            max_Area = 8000*np.exp(-1300*np.array(density[x, y]))+200
            # print(mu_Area)
            # print(var_Area)
            # print(max_Area)
            im_add, con_add = rand_nucleus(x=x, y=y, nsize = nsize, 
                                           mu_Area=mu_Area, 
                                           sigma_Area=var_Area, 
                                           max_Area = max_Area,
                                           **kwargs)
            overlap = np.sum((im_add*bin_mask)>0) / (np.sum(im_add>0).astype(np.float32))
            ntry = 0
            while (overlap>0.05) and (ntry<ntry_max):
                im_add, con_add = rand_nucleus(x=x, y=y, nsize = nsize, 
                                               mu_Area=mu_Area, 
                                               sigma_Area=var_Area, 
                                               max_Area = max_Area,
                                               **kwargs)
                overlap = np.sum((im_add*bin_mask)>0) / (np.sum(im_add>0).astype(np.float32))
                ntry += 1
            color_mask += im_add * np.random.rand(3)[None,None]
            bin_mask += im_add
            contour += con_add
            pbar.update(1)
    return color_mask, bin_mask, contour


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


def get_rand_polygon_param(s,
                           mu_Area=600, sigma_Area=400, max_Area = 500,
                           mu_ecc = 0.75, sigma_ecc = 0.24,
                           mu_irr=0.6, sigma_irr=2.0, 
                           mu_spike=0.04, sigma_spike=0.005, 
                           mu_nverti=20, sigma_nverti=8):
    x, y = int(random.random()*s), int(random.random()*s);
    # beta_mean = mu_Area
    # beta_var = sigma_Area
    beta_mean = mu_Area/max_Area
    beta_var = sigma_Area/max_Area**2
    beta_a = beta_mean**2*(1-beta_mean)/beta_var-beta_mean
    beta_b = beta_mean*(1-beta_mean)**2/beta_var-1+beta_mean
    if beta_a < 0 or beta_b < 0:
        import pdb
        pdb.set_trace()
    area = beta.rvs(beta_a, beta_b)*max_Area;
    
    ecc = (random.random()*2-1)*sigma_ecc + mu_ecc;
    irr = mu_irr+random.random()*sigma_irr;
    spike = mu_spike+random.random()*sigma_spike;
    nverti = int(random.random()*sigma_nverti+mu_nverti);
    return x, y, area, ecc, irr, spike, nverti;


def draw_polygon(x, y, area, ecc, irr, spike, nverti, s):
    vertices = generatePolygon(x, y, area, ecc, irr, spike, nverti);
    mask = Image.fromarray(np.zeros((s, s, 3), dtype=np.uint8));
    draw = ImageDraw.Draw(mask);
    draw.polygon(vertices, fill=(1,1,1));
    return np.array(mask);

def rand_nucleus(x=None, y=None, nsize = 460, **kwargs):
    rand_x, rand_y, area, ecc, irr, spike, nverti = get_rand_polygon_param(nsize, **kwargs);
    if x is None:
        x = rand_x
    if y is None:
        y = rand_y
    mask = draw_polygon(x, y, area, ecc, irr, spike, nverti, nsize);
    dx = ndimage.sobel(mask[...,0], 0)
    dy = ndimage.sobel(mask[...,0], 1)
    contour = ((dx>0) + (dy>0))[:,:,np.newaxis] * (mask==0);
    return mask, contour


def generatePolygon(ctrX, ctrY, Area, eccentricity, irregularity, spikeyness, numVerts):
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
    angle = random.uniform(0, 2*math.pi)
    init_angle = angle

    for i in range(numVerts):
        r_i = semi_minor//math.sqrt(1-(eccentricity*math.cos(angle))**2)
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
