"""Independent numerical oracles and TV1 hand calculations."""
import sys
from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from scipy.signal import convolve2d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deconv.core import convolve, gaussian_psf, save_tiff, kernel_grid, degrade
from deconv.solvers import solve, tikhonov_spectral, tv_divergence
from deconv.metrics import metrics

class HandExamples(unittest.TestCase):
    def test_convolution_1d(self) -> None:
        for peak in (8,12):
            got = convolve(np.array([[0,0,peak,0,0.]]), np.array([[.25,.5,.25]]))
            np.testing.assert_allclose(got, [[0,peak/4,peak/2,peak/4,0]], atol=1e-12)

    def test_convolution_2d(self) -> None:
        a = np.zeros((5,5)); a[2,2]=9
        expected = np.zeros_like(a); expected[1:4,1:4]=1
        np.testing.assert_allclose(convolve(a,np.ones((3,3))/9),expected,atol=1e-12)

    def test_nif_matrix(self) -> None:
        for g, expected in [([6,2],[8,0]),([6.1,1.9],[8.2,-.2])]:
            np.testing.assert_allclose(solve(np.array([g]), np.array([[.25,.75]]),'NIF'),[expected],atol=1e-12)

    def test_tikhonov_scalar(self) -> None:
        for h,g,lam,expected in [(.1,.45,0,4.5),(.1,.45,.001,4.0909090909),
                                  (.1,.45,.01,2.25),(.2,.9,.01,3.6),(.2,.9,.04,2.25)]:
            self.assertAlmostEqual(float(tikhonov_spectral(np.array(g),np.array(h),lam)),expected,places=9)

    def test_rl_two_iterations(self) -> None:
        for k,expected in [(1,[5,3]),(2,[40/7,16/7])]:
            np.testing.assert_allclose(solve(np.array([[6.,2]]),np.array([[.25,.75]]),'RL',iterations=k),[expected],atol=1e-10)

    def test_laplacian(self) -> None:
        lap=np.array([[0,1,0],[1,-4,1],[0,1,0]])
        a=np.full((5,5),5.0)
        from scipy.fft import fft2, ifft2
        def apply(x: np.ndarray) -> np.ndarray:
            return ifft2(fft2(x)*fft2(kernel_grid(lap,x.shape))).real
        np.testing.assert_allclose(apply(a),0,atol=1e-12)
        a[2,2]=9
        self.assertAlmostEqual(apply(a)[2,2],-16)

    def test_psnr_hand(self) -> None:
        for error, expected in [(.1,20),(np.sqrt(.001),30)]:
            self.assertAlmostEqual(metrics(np.zeros((11,11)),np.full((11,11),error))['psnr'],expected)

class Properties(unittest.TestCase):
    def test_impulse_even_odd(self) -> None:
        for size in (16,17,18):
            h=gaussian_psf(size); self.assertAlmostEqual(h.sum(),1)
            a=np.zeros((65,64)); a[32,32]=1
            b=convolve(a,h)
            self.assertEqual(np.unravel_index(b.argmax(),b.shape),(32,32))
            self.assertAlmostEqual(b.sum(),1)

    def test_spatial_oracle_asymmetric_psf(self) -> None:
        a=np.random.default_rng(1).random((17,19))
        h=np.array([[1,2,3],[0,4,1],[2,0,7]],float);h/=h.sum()
        np.testing.assert_allclose(convolve(a,h),convolve2d(a,h,mode='same',boundary='wrap'),atol=1e-12)

    def test_nif_identity(self) -> None:
        a=np.random.default_rng(2).random((17,19))
        np.testing.assert_allclose(solve(a,np.ones((1,1)),'NIF'),a,atol=1e-12)

    def test_dense_trif_rif(self) -> None:
        shape=(5,5); n=25
        h=np.array([[0,1,0],[1,4,1],[0,1,0]],float)/8
        lap=np.array([[0,1,0],[1,-4,1],[0,1,0]])
        basis=np.eye(n).reshape(n,*shape)
        H=np.column_stack([convolve2d(b,h,'same',boundary='wrap').ravel() for b in basis])
        L=np.column_stack([convolve2d(b,lap,'same',boundary='wrap').ravel() for b in basis])
        g=np.random.default_rng(3).random(shape)
        for method,regularizer in [('TRIF',np.eye(n)),('RIF',L.T@L)]:
            expected=np.linalg.solve(H.T@H+.02*regularizer,H.T@g.ravel()).reshape(shape)
            np.testing.assert_allclose(solve(g,h,method,lam=.02),expected,atol=1e-11)

    def test_rif_laplacian_for_single_row(self) -> None:
        g = np.array([[.8, .2, .6, .1, .5]])
        eye = np.eye(g.size)
        lap = np.roll(eye, 1, axis=0) + np.roll(eye, -1, axis=0) - 2 * eye
        expected = np.linalg.solve(eye + .02 * lap.T @ lap, g.ravel())
        np.testing.assert_allclose(solve(g, np.ones((1, 1)), 'RIF', lam=.02), [expected],
                                   atol=1e-12)

    def test_baseline_equivalence(self) -> None:
        for shape in [(31,29),(32,34)]:
            g=np.random.default_rng(4).random(shape);h=gaussian_psf(7)
            for method in ['NIF','TRIF','RIF','RL','RLTV']:
                np.testing.assert_allclose(solve(g,h,method,lam=.001,iterations=3),
                                           solve(g,h,method,lam=.001,iterations=3,optimized=False),atol=1e-10)

    def test_rl_adjoint_asymmetric(self) -> None:
        g=np.random.default_rng(5).random((9,11))
        h=np.array([[1,2,0],[3,1,4],[1,0,2]],float);h/=h.sum()
        f=np.full_like(g,g.mean())
        for _ in range(2):
            f*=convolve2d(g/(convolve2d(f,h,'same',boundary='wrap')+1e-12),h[::-1,::-1],'same',boundary='wrap')
        np.testing.assert_allclose(solve(g,h,'RL',iterations=2),f,atol=1e-12)

    def test_rl_imagej_rule(self) -> None:
        g = np.array([[0., .02, .4, .8, .01]])
        h = np.array([[.1, .7, .2]])
        prediction = convolve2d(g, h, 'same', boundary='wrap')
        ratio = np.divide(g, prediction, out=np.zeros_like(g), where=prediction >= .15)
        expected = g * convolve2d(ratio, h[:, ::-1], 'same', boundary='wrap')
        actual = solve(g, h, 'RL', iterations=1, eps=.15, imagej_rl=True)
        np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_tv_gradient(self) -> None:
        f=np.random.default_rng(6).random((7,9)); direction=np.random.default_rng(7).normal(size=f.shape)
        def tv(x: np.ndarray) -> float:
            return float(np.sqrt((np.roll(x,-1,0)-x)**2+(np.roll(x,-1,1)-x)**2+1e-12).sum())
        eps=1e-6
        self.assertAlmostEqual((tv(f+eps*direction)-tv(f-eps*direction))/(2*eps),float((-tv_divergence(f,delta=1e-6)*direction).sum()),places=6)

    def test_rltv_zero_and_constant(self) -> None:
        g=np.random.default_rng(8).random((17,19));h=gaussian_psf(7)
        np.testing.assert_allclose(solve(g,h,'RL'),solve(g,h,'RLTV',lam=0))
        for method in ['RL','RLTV']:
            out=solve(g,h,method,lam=.001)
            self.assertTrue(np.isfinite(out).all() and (out>=0).all())
            np.testing.assert_allclose(solve(np.ones((17,19)),h,method,lam=.001),1,atol=1e-10)
            np.testing.assert_equal(solve(np.zeros((17,19)),h,method,lam=.001),0)

    def test_tiff_raw_roundtrip(self) -> None:
        a=np.random.default_rng(9).normal(size=(17,19))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'test.tif';save_tiff(path,a)
            with Image.open(path) as im:
                self.assertEqual(im.tag_v2[259],1)
                np.testing.assert_allclose(np.asarray(im),a,atol=2e-7,rtol=0)

    def test_ssim_independent_single_window(self) -> None:
        a=np.random.default_rng(10).random((11,11));b=np.random.default_rng(11).random((11,11))
        axis=np.arange(-5,6);w=np.exp(-(axis[:,None]**2+axis[None,:]**2)/(2*1.5**2));w/=w.sum()
        ma,mb=(w*a).sum(),(w*b).sum()
        va,vb=(w*(a-ma)**2).sum(),(w*(b-mb)**2).sum();cov=(w*(a-ma)*(b-mb)).sum()
        expected=(2*ma*mb+.0001)*(2*cov+.0009)/((ma*ma+mb*mb+.0001)*(va+vb+.0009))
        self.assertAlmostEqual(metrics(a,b)['ssim'],expected)
        self.assertAlmostEqual(metrics(a,a)['ssim'],1)
        self.assertTrue(np.isinf(metrics(a,a)['psnr']))

    def test_seed_and_inputs(self) -> None:
        a=np.ones((17,19))*.4;h=gaussian_psf(7)
        np.testing.assert_equal(degrade(a,h,'poisson',42)[1],degrade(a,h,'poisson',42)[1])
        for g,k,method,kwargs in [(a,-h,'RL',{}),(-a,h,'RL',{}),(a,h,'bad',{}),
                                  (a,h,'TRIF',{'lam':-1}),(a,h,'RL',{'iterations':0})]:
            with self.assertRaises(ValueError): solve(g,k,method,**kwargs)

if __name__=='__main__': unittest.main()
