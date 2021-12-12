from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from ot.utils import proj_simplex
from ot.datasets import make_1D_gauss as gauss
from ot.lp import wasserstein_1d
import torch
import matplotlib as mpl
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pylab as pl
import ot
import ot.plot
from django.shortcuts import render
mpl.use('Agg')

############################################################################################

# =================================
# OT and regularized OT
# =================================

# parameters

n = 100  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
a = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
b = ot.datasets.make_1D_gauss(n, m=60, s=10)

# loss matrix
M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
M /= M.max()

# Solve EMD
############################################################################################

# EMD

G0 = ot.emd(a, b, M)

pl.figure(3, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, G0, 'OT matrix G0')

imgD = BytesIO()

plt.savefig(imgD, format='png')
plt.close()
imgD.seek(0)
plot_url_D = base64.b64encode(imgD.getvalue()).decode('utf8')


# Solve EMD with Frobenius norm regularization
############################################################################################

# Example with Frobenius norm regularization


def f(G):
    return 0.5 * np.sum(G**2)


def df(G):
    return G


reg = 1e-1

Gl2 = ot.optim.cg(a, b, M, reg, f, df, verbose=True)

pl.figure(3)
ot.plot.plot1D_mat(a, b, Gl2, 'OT matrix Frob. reg')

imgC = BytesIO()

plt.savefig(imgC, format='png')
plt.close()
imgC.seek(0)
plot_url_C = base64.b64encode(imgC.getvalue()).decode('utf8')


# Solve EMD with entropic regularization
############################################################################################

# Example with entropic regularization


def f(G):
    return np.sum(G * np.log(G))


def df(G):
    return np.log(G) + 1.


reg = 1e-3

Ge = ot.optim.cg(a, b, M, reg, f, df, verbose=True)

pl.figure(4, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Ge, 'OT matrix Entrop. reg')

imgB = BytesIO()

plt.savefig(imgB, format='png')
plt.close()
imgB.seek(0)
plot_url_B = base64.b64encode(imgB.getvalue()).decode('utf8')


# Solve EMD with Frobenius norm + entropic regularization
############################################################################################

# Example with Frobenius norm + entropic regularization with gcg


def f(G):
    return 0.5 * np.sum(G**2)


def df(G):
    return G


reg1 = 1e-3
reg2 = 1e-1

Gel2 = ot.optim.gcg(a, b, M, reg1, reg2, f, df, verbose=True)

pl.figure(5, figsize=(5, 5))
ot.plot.plot1D_mat(a, b, Gel2, 'OT entropic + matrix Frob. reg')

imgA = BytesIO()

plt.savefig(imgA, format='png')
plt.close()
imgA.seek(0)
plot_url_A = base64.b64encode(imgA.getvalue()).decode('utf8')
# pl.show()

############################################################################################

# =================================
# POT backend
# =================================


red = np.array(mpl.colors.to_rgb('red'))
blue = np.array(mpl.colors.to_rgb('blue'))

############################################################################################

n = 100  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
a = gauss(n, m=20, s=5)  # m= mean, s= std
b = gauss(n, m=60, s=10)

# enforce sum to one on the support
a = a / a.sum()
b = b / b.sum()


device = "cuda" if torch.cuda.is_available() else "cpu"

# use pyTorch for our data
x_torch = torch.tensor(x).to(device=device)
a_torch = torch.tensor(a).to(device=device).requires_grad_(True)
b_torch = torch.tensor(b).to(device=device)

lr = 1e-6
nb_iter_max = 800

loss_iter = []

pl.figure(1, figsize=(8, 4))
pl.plot(x, a, 'b', label='Source distribution')
pl.plot(x, b, 'r', label='Target distribution')

for i in range(nb_iter_max):
    # Compute the Wasserstein 1D with torch backend
    loss = wasserstein_1d(x_torch, x_torch, a_torch, b_torch, p=2)
    # record the corresponding loss value
    loss_iter.append(loss.clone().detach().cpu().numpy())
    loss.backward()

    # performs a step of projected gradient descent
    with torch.no_grad():
        grad = a_torch.grad
        a_torch -= a_torch.grad * lr  # step
        a_torch.grad.zero_()
        a_torch.data = proj_simplex(a_torch)  # projection onto the simplex

    # plot one curve every 10 iterations
    if i % 10 == 0:
        mix = float(i) / nb_iter_max
        pl.plot(x, a_torch.clone().detach().cpu().numpy(),
                c=(1 - mix) * blue + mix * red)

pl.legend()
pl.title('Distribution along the iterations of the projected gradient descent')
# pl.show()
imageA = BytesIO()

plt.savefig(imageA, format='png')
plt.close()
imageA.seek(0)
plotUrl_A = base64.b64encode(imageA.getvalue()).decode('utf8')


pl.figure(2)
pl.plot(range(nb_iter_max), loss_iter, lw=3)
pl.title('Evolution of the loss along iterations', fontsize=16)
# pl.show()
imageB = BytesIO()

plt.savefig(imageB, format='png')
plt.close()
imageB.seek(0)
plotUrl_B = base64.b64encode(imageB.getvalue()).decode('utf8')

############################################################################################

# Wasserstein barycenter

device = "cuda" if torch.cuda.is_available() else "cpu"

# use pyTorch for our data
x_torch = torch.tensor(x).to(device=device)
a_torch = torch.tensor(a).to(device=device)
b_torch = torch.tensor(b).to(device=device)
bary_torch = torch.tensor(
    (a + b).copy() / 2).to(device=device).requires_grad_(True)


lr = 1e-6
nb_iter_max = 2000

loss_iter = []

# instant of the interpolation
t = 0.5

for i in range(nb_iter_max):
    # Compute the Wasserstein 1D with torch backend
    loss = (1 - t) * wasserstein_1d(x_torch, x_torch, a_torch.detach(), bary_torch,
                                    p=2) + t * wasserstein_1d(x_torch, x_torch, b_torch, bary_torch, p=2)
    # record the corresponding loss value
    loss_iter.append(loss.clone().detach().cpu().numpy())
    loss.backward()

    # performs a step of projected gradient descent
    with torch.no_grad():
        grad = bary_torch.grad
        bary_torch -= bary_torch.grad * lr  # step
        bary_torch.grad.zero_()
        # projection onto the simplex
        bary_torch.data = proj_simplex(bary_torch)

pl.figure(3, figsize=(8, 4))
pl.plot(x, a, 'b', label='Source distribution')
pl.plot(x, b, 'r', label='Target distribution')
pl.plot(x, bary_torch.clone().detach().cpu().numpy(),
        c='green', label='W barycenter')
pl.legend()
pl.title('Wasserstein barycenter computed by gradient descent')
# pl.show()

imageC = BytesIO()

plt.savefig(imageC, format='png')
plt.close()
imageC.seek(0)
plotUrl_C = base64.b64encode(imageC.getvalue()).decode('utf8')

pl.figure(4)
pl.plot(range(nb_iter_max), loss_iter, lw=3)
pl.title('Evolution of the loss along iterations', fontsize=16)
# pl.show()

imageD = BytesIO()

plt.savefig(imageD, format='png')
plt.close()
imageD.seek(0)
plotUrl_D = base64.b64encode(imageD.getvalue()).decode('utf8')


############################################################################################
# ==============================
# 1D Wasserstein barycenter demo
# ==============================

# Generate data

# parameters

n = 100  # nb bins

# bin positions
x = np.arange(n, dtype=np.float64)

# Gaussian distributions
a1 = ot.datasets.make_1D_gauss(n, m=20, s=5)  # m= mean, s= std
a2 = ot.datasets.make_1D_gauss(n, m=60, s=8)

# creating matrix A containing all distributions
A = np.vstack((a1, a2)).T
n_distributions = A.shape[1]

# loss matrix + normalization
M = ot.utils.dist0(n)
M /= M.max()

##############################################################################
# Barycenter computation

# barycenter computation

alpha = 0.2  # 0<=alpha<=1
weights = np.array([1 - alpha, alpha])

# l2bary
bary_l2 = A.dot(weights)

# wasserstein
reg = 1e-3
bary_wass = ot.bregman.barycenter(A, M, reg, weights)

f, (ax1, ax2) = plt.subplots(2, 1, tight_layout=True, num=1)
ax1.plot(x, A, color="black")
ax1.set_title('Distributions')

ax2.plot(x, bary_l2, 'r', label='l2')
ax2.plot(x, bary_wass, 'g', label='Wasserstein')
ax2.set_title('Barycenters')

plt.legend()
# plt.show()
img_A = BytesIO()

plt.savefig(img_A, format='png')
plt.close()
img_A.seek(0)
plotUrlA = base64.b64encode(img_A.getvalue()).decode('utf8')


##############################################################################
# Barycentric interpolation

#  barycenter interpolation

n_alpha = 11
alpha_list = np.linspace(0, 1, n_alpha)


B_l2 = np.zeros((n, n_alpha))

B_wass = np.copy(B_l2)

for i in range(n_alpha):
    alpha = alpha_list[i]
    weights = np.array([1 - alpha, alpha])
    B_l2[:, i] = A.dot(weights)
    B_wass[:, i] = ot.bregman.barycenter(A, M, reg, weights)

# plot interpolation
plt.figure(2)

cmap = plt.cm.get_cmap('viridis')
verts = []
zs = alpha_list
for i, z in enumerate(zs):
    ys = B_l2[:, i]
    verts.append(list(zip(x, ys)))

ax = plt.gcf().gca(projection='3d')

poly = PolyCollection(verts, facecolors=[cmap(a) for a in alpha_list])
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')
ax.set_xlabel('x')
ax.set_xlim3d(0, n)
ax.set_ylabel('$\\alpha$')
ax.set_ylim3d(0, 1)
ax.set_zlabel('')
ax.set_zlim3d(0, B_l2.max() * 1.01)
plt.title('Barycenter interpolation with l2')
plt.tight_layout()

plt.figure(3)
cmap = plt.cm.get_cmap('viridis')
verts = []
zs = alpha_list
for i, z in enumerate(zs):
    ys = B_wass[:, i]
    verts.append(list(zip(x, ys)))

ax = plt.gcf().gca(projection='3d')

poly = PolyCollection(verts, facecolors=[cmap(a) for a in alpha_list])
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=zs, zdir='y')
ax.set_xlabel('x')
ax.set_xlim3d(0, n)
ax.set_ylabel('$\\alpha$')
ax.set_ylim3d(0, 1)
ax.set_zlabel('')
ax.set_zlim3d(0, B_l2.max() * 1.01)
plt.title('Barycenter interpolation with Wasserstein')
plt.tight_layout()

img_B = BytesIO()

plt.savefig(img_B, format='png')
plt.close()
img_B.seek(0)
plotUrlB = base64.b64encode(img_B.getvalue()).decode('utf8')


############################################################################################

# ======================================================
# OT with Laplacian regularization for domain adaptation
# ======================================================

##############################################################################
# Generate data

n_source_samples = 150
n_target_samples = 150

Xs, ys = ot.datasets.make_data_classif('3gauss', n_source_samples)
Xt, yt = ot.datasets.make_data_classif('3gauss2', n_target_samples)


##############################################################################
# Instantiate the different transport algorithms and fit them

# EMD Transport
ot_emd = ot.da.EMDTransport()
ot_emd.fit(Xs=Xs, Xt=Xt)

# Sinkhorn Transport
ot_sinkhorn = ot.da.SinkhornTransport(reg_e=.01)
ot_sinkhorn.fit(Xs=Xs, Xt=Xt)

# EMD Transport with Laplacian regularization
ot_emd_laplace = ot.da.EMDLaplaceTransport(reg_lap=100, reg_src=1)
ot_emd_laplace.fit(Xs=Xs, Xt=Xt)

# transport source samples onto target samples
transp_Xs_emd = ot_emd.transform(Xs=Xs)
transp_Xs_sinkhorn = ot_sinkhorn.transform(Xs=Xs)
transp_Xs_emd_laplace = ot_emd_laplace.transform(Xs=Xs)

##############################################################################
# Fig 1 : plots source and target samples

pl.figure(1, figsize=(10, 5))
pl.subplot(1, 2, 1)
pl.scatter(Xs[:, 0], Xs[:, 1], c=ys, marker='+', label='Source samples')
pl.xticks([])
pl.yticks([])
pl.legend(loc=0)
pl.title('Source  samples')

pl.subplot(1, 2, 2)
pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o', label='Target samples')
pl.xticks([])
pl.yticks([])
pl.legend(loc=0)
pl.title('Target samples')
pl.tight_layout()

imgS = BytesIO()
plt.savefig(imgS, format='png')
plt.close()
imgS.seek(0)
plotUrlS = base64.b64encode(imgS.getvalue()).decode('utf8')


##############################################################################
# Fig 2 : plot optimal couplings and transported samples

param_img = {'interpolation': 'nearest'}

pl.figure(2, figsize=(15, 8))
pl.subplot(2, 3, 1)
pl.imshow(ot_emd.coupling_, **param_img)
pl.xticks([])
pl.yticks([])
pl.title('Optimal coupling\nEMDTransport')

pl.figure(2, figsize=(15, 8))
pl.subplot(2, 3, 2)
pl.imshow(ot_sinkhorn.coupling_, **param_img)
pl.xticks([])
pl.yticks([])
pl.title('Optimal coupling\nSinkhornTransport')

pl.subplot(2, 3, 3)
pl.imshow(ot_emd_laplace.coupling_, **param_img)
pl.xticks([])
pl.yticks([])
pl.title('Optimal coupling\nEMDLaplaceTransport')

pl.subplot(2, 3, 4)
pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=0.3)
pl.scatter(transp_Xs_emd[:, 0], transp_Xs_emd[:, 1], c=ys,
           marker='+', label='Transp samples', s=30)
pl.xticks([])
pl.yticks([])
pl.title('Transported samples\nEmdTransport')
pl.legend(loc="lower left")

pl.subplot(2, 3, 5)
pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=0.3)
pl.scatter(transp_Xs_sinkhorn[:, 0], transp_Xs_sinkhorn[:, 1], c=ys,
           marker='+', label='Transp samples', s=30)
pl.xticks([])
pl.yticks([])
pl.title('Transported samples\nSinkhornTransport')

pl.subplot(2, 3, 6)
pl.scatter(Xt[:, 0], Xt[:, 1], c=yt, marker='o',
           label='Target samples', alpha=0.3)
pl.scatter(transp_Xs_emd_laplace[:, 0], transp_Xs_emd_laplace[:, 1], c=ys,
           marker='+', label='Transp samples', s=30)
pl.xticks([])
pl.yticks([])
pl.title('Transported samples\nEMDLaplaceTransport')
pl.tight_layout()

# pl.show()

imgTS = BytesIO()
plt.savefig(imgTS, format='png')
plt.close()
imgTS.seek(0)
plotUrlTS = base64.b64encode(imgTS.getvalue()).decode('utf8')

############################################################################################

# view on http://127.0.0.1:8000/


def home(request):
    return render(request, 'home.html', {"plot_url_A": plot_url_A, "plot_url_B": plot_url_B, "plot_url_C": plot_url_C, "plot_url_D": plot_url_D, "plotUrl_D": plotUrl_D, "plotUrl_C": plotUrl_C, "plotUrl_B": plotUrl_B, "plotUrl_A": plotUrl_A, "plotUrlTS": plotUrlTS, "plotUrlS": plotUrlS, "plotUrlB": plotUrlB, "plotUrlA": plotUrlA})
