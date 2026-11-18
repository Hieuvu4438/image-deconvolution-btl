import ij.IJ;
import ij.ImagePlus;
import ij.Macro;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Locale;

/** Runs the ImageJ plugin on float TIFFs and independently scores its saved output. */
public final class BatchReference {
    private static ImagePlus openFloat(String path) {
        ImagePlus image = IJ.openImage(path);
        if (image == null || image.getBitDepth() != 32 || image.getStackSize() != 1) {
            throw new IllegalArgumentException("Expected single-slice float TIFF: " + path);
        }
        return image;
    }

    private static String optionsPath(String path) {
        return path.replace('\\', '/');
    }

    private static signal.RealSignal signalOf(ImagePlus image) {
        signal.RealSignal result = new signal.RealSignal(image.getTitle(), image.getWidth(), image.getHeight(), 1);
        result.data[0] = (float[]) image.getProcessor().getPixelsCopy();
        return result;
    }

    private static void score(PrintWriter csv, String id, String method, ImagePlus truth,
                              ImagePlus result, double nativePsnr) {
        int width = truth.getWidth(), height = truth.getHeight();
        if (result.getWidth() != width || result.getHeight() != height || width < 11 || height < 11) {
            throw new IllegalArgumentException("Incompatible image dimensions: " + id);
        }
        float[] ref = (float[]) truth.getProcessor().getPixels();
        float[] est = (float[]) result.getProcessor().getPixels();
        double[] window = new double[11];
        double weight = 0;
        for (int d = -5; d <= 5; d++) {
            window[d + 5] = Math.exp(-d * d / 4.5);
            weight += window[d + 5];
        }
        for (int d = 0; d < 11; d++) window[d] /= weight;

        for (String policy : new String[] {"raw", "clip01"}) {
            double[] pixels = new double[est.length];
            double mse = 0;
            for (int i = 0; i < est.length; i++) {
                if (!Float.isFinite(est[i]) || !Float.isFinite(ref[i])) {
                    throw new IllegalArgumentException("Non-finite pixel: " + id);
                }
                pixels[i] = policy.equals("raw") ? est[i] : Math.max(0, Math.min(1, est[i]));
                double error = pixels[i] - ref[i];
                mse += error * error;
            }
            mse /= pixels.length;

            double ssim = 0;
            for (int y = 5; y < height - 5; y++) {
                for (int x = 5; x < width - 5; x++) {
                    double ux = 0, uy = 0, exx = 0, eyy = 0, exy = 0;
                    for (int dy = -5; dy <= 5; dy++) {
                        for (int dx = -5; dx <= 5; dx++) {
                            int i = (y + dy) * width + x + dx;
                            double w = window[dy + 5] * window[dx + 5];
                            double a = ref[i], b = pixels[i];
                            ux += w * a;
                            uy += w * b;
                            exx += w * a * a;
                            eyy += w * b * b;
                            exy += w * a * b;
                        }
                    }
                    double vx = Math.max(0, exx - ux * ux), vy = Math.max(0, eyy - uy * uy);
                    double cov = exy - ux * uy;
                    ssim += (2 * ux * uy + .0001) * (2 * cov + .0009)
                          / ((ux * ux + uy * uy + .0001) * (vx + vy + .0009));
                }
            }
            ssim /= (width - 10) * (height - 10);
            csv.printf(Locale.ROOT, "%s,%s,%s,%.15g,%.15g,%.15g,%s%n",
                       id, method, policy, mse, -10 * Math.log10(mse), ssim,
                       policy.equals("raw") ? Double.toString(nativePsnr) : "");
        }
    }

    public static void main(String[] args) throws Exception {
        Path jobs = Paths.get(args[0]).toAbsolutePath();
        Path logDir = jobs.getParent();
        Path pluginDir = Files.createDirectories(logDir.resolve("plugins"));
        System.setProperty("plugins.dir", pluginDir.getParent().toString());
        Thread.currentThread().setName("Run$_DeconvolutionBatch");
        System.out.println("ImageJ " + IJ.getVersion() + "; Java " + System.getProperty("java.version"));
        try (PrintWriter csv = new PrintWriter(logDir.resolve("imagej_metrics.csv").toFile(), "UTF-8")) {
            csv.println("case,method,policy,mse,psnr,ssim,dl2_psnr_raw");
            for (String row : Files.readAllLines(jobs)) {
                String[] job = row.split("\t", -1);
                String id = job[0], method = job[1];
                ImagePlus truth = openFloat(job[2]);
                Path dest = Paths.get(job[5]);
                String result = job[3];
                if (!method.equals("observed")) {
                    Files.createDirectories(dest);
                    Path tif = dest.resolve(method + ".tif");
                    if (Files.exists(tif)) throw new IllegalStateException("Output already exists: " + tif);
                    String command = "-image file " + optionsPath(job[3]) + " -psf file " + optionsPath(job[4])
                            + " -algorithm " + job[6] + " -epsilon " + job[7]
                            + " -pad NO NO 0 0 -apo NO NO -norm " + job[8] + " -fft Academic"
                            + " -monitor console -stats false -constraint no -multithreading no"
                            + " -path " + optionsPath(dest.toString()) + " -out stack " + method + " intact float noshow";
                    System.out.println(id + ": " + command);
                    Macro.setOptions(command);
                    new DeconvolutionLab2_Run().run("");
                    Macro.setOptions(null);
                    result = tif.toString();
                }
                ImagePlus estimate = openFloat(result);
                signal.RealSignal actual = signalOf(estimate);
                signal.RealSignal ideal = signalOf(truth);
                double nativePsnr = signal.Assessment.psnr(actual, ideal);
                signal.SignalCollector.free(actual);
                signal.SignalCollector.free(ideal);
                score(csv, id, method, truth, estimate, nativePsnr);
                truth.close();
                estimate.close();
                csv.flush();
            }
        }
        System.exit(0);
    }
}
