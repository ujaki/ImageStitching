using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using System.Windows.Forms;
using System.Diagnostics;
using Emgu.CV;
using Emgu.CV.Cuda;
using Emgu.CV.XFeatures2D;
using Emgu.CV.Stitching;
using Emgu.CV.Structure;
using Emgu.CV.CvEnum;
using Emgu.CV.Util;
using Emgu.CV.Features2D;

/*
 GpuMat g; 
Mat m; 
1. g = GpuMat(m) -> takes around 2 ms 
2. g.upload(m)   -> takes roughly 1 ms.
     */

namespace StitchingProgram
{
    public partial class Form1 : Form
    {
        
        private int numOfCameras = 2;
        VideoCapture[] cameras;
        public Mat[] imgs;
        public GpuMat gpuMats1;
        public GpuMat gpuMats2;
        private int fps = 30;


        public Form1()
        {
            InitializeComponent();
            InitializationOfCameras(); //카메라 설정 초기화
        }

        private void InitializationOfCameras()
        {
            //640, 480
            imgs = new Mat[numOfCameras];
           // gpuMats = new GpuMat[numOfCameras];
            cameras = new VideoCapture[numOfCameras];
            for (int i = 0; i < numOfCameras; i++)
            {
                cameras[i] = new VideoCapture(i);
                cameras[i].SetCaptureProperty(CapProp.Fps, fps);
                cameras[i].SetCaptureProperty(CapProp.FrameWidth, 1920);
                cameras[i].SetCaptureProperty(CapProp.FrameHeight, 1080);
            }

            Application.Idle += GetFrameFromCameras;
           // new Thread(new ThreadStart(GetFrameFromCameras)).Start();
        
        }

        private void GetFrameFromCameras(Object sender, EventArgs e)
        {
                try
                {
                    //MessageBox.Show("sfasdadasd");
                    for (int i = 0; i < numOfCameras; i++)
                    {
                        imgs[i] = cameras[i].QueryFrame();
                    //cameras[i].
                    
                      //  gpuMats[i].Upload(imgs[0]);
                        // gpuMats[i].Upload(imgs[i]);
                    }
                // imgs[0].ToImage<Gray, byte>();
               
                imageBox1.Image = imgs[0];
                imageBox2.Image = imgs[1];
                gpuMats1 = new GpuMat();
                gpuMats2 = new GpuMat();
                //new Thread(new ThreadStart(Stitching)).Start();
                //  Mat result = Draw(imgs[0], imgs[1], out time);
                //  imageBox3.Image = result;
                // Thread.Sleep(1000 / fps);
                //await Task.Delay(1000 / fps);
            }
                catch (Exception exception)
                {
                    MessageBox.Show(exception.ToString());
                }
        }
        /*
        private void Stitching()
        {
            Debug.WriteLine(CudaInvoke.HasCuda.ToString());
            using (Stitcher stitcher = new Stitcher(CudaInvoke.HasCuda))
            {
              //  stitcher.SetWarper(new PlaneWarper(1f));
                    
                    
                    
                    //stitcher.SetFeaturesFinder(finder);
                    Debug.WriteLine(String.Format("PanoConfie : {0}", stitcher.PanoConfidenceThresh.ToString()));
                    using (VectorOfMat mat = new VectorOfMat())
                    {
                        while (true)
                        {
                            try
                            {
                                if (imgs[0] != null && imgs[1] != null)
                                {
                                    Mat resultImage = new Mat();
                                    mat.Push(imgs[0]);
                                    mat.Push(imgs[1]);
                                //gpuMats1.Upload(imgs[0], new Stream());
                                //gpuMats2.Upload(imgs[1], new Stream());

                                
                                imageBox3.Image = resultImage;
                                //mat.Push(gpuMats1);
                                //mat.Push(gpuMats2);
                                Stitcher.Status stitchStatus = stitcher.Stitch(mat, resultImage);
                                    // Debug.WriteLine(stitcher.SeamEstimationResol.ToString());
                                    if (stitchStatus == Stitcher.Status.Ok)
                                    {
                                    // CvInvoke.Imshow("Stitched Image", resultImage);
                                  
                                        
                                        continue;
                                        //imageBox3.BeginInvoke(new Action(() => { ResultImageBox.Image = resultImage; }));
                                        Debug.WriteLine(String.Format("Stitching Status : {0}", stitchStatus));

                                    }
                                    else
                                    {
                                        //MessageBox.Show(this, String.Format("Stiching Error: {0}", stitchStatus));
                                        //imageBoxResult.BeginInvoke(new Action(() => { imageBoxResult.Image = null; }));

                                        Debug.WriteLine(String.Format("Stiching Error: {0}", stitchStatus));
                                        continue;
                                    }
                                }
                            }
                            catch (AccessViolationException exception)
                            {
                                MessageBox.Show(exception.ToString());
                                return;
                            }
                        }
                    }
                
            }
        }
        */

        private void button1_Click(object sender, EventArgs e)
        {
            new Thread(new ThreadStart(Stitching)).Start();
        }

        private void FeatureDectecting()
        {
            long match_time;
            Mat result_mat = Draw(imgs[0], imgs[1], out match_time);
            imageBox3.Image = result_mat;
        }
        public static void FindMatch(Mat modelImage, Mat observedImage, out long matchTime, out VectorOfKeyPoint modelKeyPoints,
                                    out VectorOfKeyPoint observedKeyPoints, VectorOfVectorOfDMatch matches, out Mat mask, out Mat homography)
        {

            int k = 2;
            double uniquenessThreshold = 0.8;
            double hessianThresh = 300;
            matchTime = 1;
            Stopwatch watch;
            homography = null;

            modelKeyPoints = new VectorOfKeyPoint();
            observedKeyPoints = new VectorOfKeyPoint();
           
                // using (UMat uModelImage = modelImage.ToUMat(AccessType.Read))
                using (UMat uModelImage = modelImage.GetUMat(AccessType.Read))

                using (UMat uObservedImage = observedImage.GetUMat(AccessType.Read))
                {
                    SURF surfCPU = new SURF(hessianThresh);
                    //extract features from the object image
                    UMat modelDescriptors = new UMat();
                    surfCPU.DetectAndCompute(uModelImage, null, modelKeyPoints, modelDescriptors, false);

                    //watch = Stopwatch.StartNew();

                    // extract features from the observed image
                    UMat observedDescriptors = new UMat();
                    surfCPU.DetectAndCompute(uObservedImage, null, observedKeyPoints, observedDescriptors, false);
                    BFMatcher matcher = new BFMatcher(DistanceType.L2);
                    matcher.Add(modelDescriptors);
                    try
                    {
                        matcher.KnnMatch(observedDescriptors, matches, k, null);
                    }
                    catch (Exception exp)
                    {
                        MessageBox.Show(exp.Message);
                    }

                    mask = new Mat(matches.Size, 1, DepthType.Cv8U, 1);
                    mask.SetTo(new MCvScalar(255));
                    Features2DToolbox.VoteForUniqueness(matches, uniquenessThreshold, mask);

                    int nonZeroCount = CvInvoke.CountNonZero(mask);
                    if (nonZeroCount >= 4)
                    {
                        nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints,
                           matches, mask, 1.5, 20);
                        if (nonZeroCount >= 4)
                            homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints,
                               observedKeyPoints, matches, mask, 2);
                    }

                   // watch.Stop();
                }
               
            
           // matchTime = watch.ElapsedMilliseconds;
        }
    

        /// <summary>
        /// Draw the model image and observed image, the matched features and homography projection.
        /// </summary>
        /// <param name="modelImage">The model image</param>
        /// <param name="observedImage">The observed image</param>
        /// <param name="matchTime">The output total time for computing the homography matrix.</param>
        /// <returns>The model image and observed image, the matched features and homography projection.</returns>
        public static Mat Draw(Mat modelImage, Mat observedImage, out long matchTime)
        {
            Mat homography;
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;

            using (VectorOfVectorOfDMatch matches = new VectorOfVectorOfDMatch())
            {
                Mat mask;
                FindMatch(modelImage, observedImage, out matchTime, out modelKeyPoints, out observedKeyPoints, matches,
                   out mask, out homography);

                //Draw the matched keypoints
                Mat result = new Mat();
                Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, observedImage, observedKeyPoints,
                   matches, result, new MCvScalar(255, 255, 255), new MCvScalar(255, 255, 255), mask);

                #region draw the projected region on the image

                if (homography != null)
                {
                    //draw a rectangle along the projected model
                    Rectangle rect = new Rectangle(Point.Empty, modelImage.Size);
                    PointF[] pts = new PointF[]
                    {
                  new PointF(rect.Left, rect.Bottom),
                  new PointF(rect.Right, rect.Bottom),
                  new PointF(rect.Right, rect.Top),
                  new PointF(rect.Left, rect.Top)
                    };
                    pts = CvInvoke.PerspectiveTransform(pts, homography);

                    Point[] points = Array.ConvertAll<PointF, Point>(pts, Point.Round);
                    using (VectorOfPoint vp = new VectorOfPoint(points))
                    {
                        CvInvoke.Polylines(result, vp, true, new MCvScalar(255, 0, 0, 255), 5);
                    }

                }

                #endregion

                return result;

            }
        }

        /*
        public static Image<Bgr, Byte> Draw(Image<Gray, Byte> modelImage, Image<Gray, byte> observedImage)
        {
            HomographyMatrix homography = null;

            FastDetector fastCPU = new FastDetector(10, true);
            VectorOfKeyPoint modelKeyPoints;
            VectorOfKeyPoint observedKeyPoints;
            Matrix<int> indices;

            BriefDescriptorExtractor descriptor = new BriefDescriptorExtractor();

            Matrix<byte> mask;
            int k = 2;
            double uniquenessThreshold = 0.8;

            //extract features from the object image
            modelKeyPoints = fastCPU.DetectKeyPointsRaw(modelImage, null);
            Matrix<Byte> modelDescriptors = descriptor.ComputeDescriptorsRaw(modelImage, null, modelKeyPoints);

            // extract features from the observed image
            observedKeyPoints = fastCPU.DetectKeyPointsRaw(observedImage, null);
            Matrix<Byte> observedDescriptors = descriptor.ComputeDescriptorsRaw(observedImage, null, observedKeyPoints);
            BruteForceMatcher<Byte> matcher = new BruteForceMatcher<Byte>(DistanceType.L2);
            matcher.Add(modelDescriptors);

            indices = new Matrix<int>(observedDescriptors.Rows, k);
            using (Matrix<float> dist = new Matrix<float>(observedDescriptors.Rows, k))
            {
                matcher.KnnMatch(observedDescriptors, indices, dist, k, null);
                mask = new Matrix<byte>(dist.Rows, 1);
                mask.SetValue(255);
                Features2DToolbox.VoteForUniqueness(dist, uniquenessThreshold, mask);
            }

            int nonZeroCount = CvInvoke.cvCountNonZero(mask);
            if (nonZeroCount >= 4)
            {
                nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, indices, mask, 1.5, 20);
                if (nonZeroCount >= 4)
                    homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(
                    modelKeyPoints, observedKeyPoints, indices, mask, 2);
            }

            //Draw the matched keypoints
            Image<Bgr, Byte> result = Features2DToolbox.DrawMatches(modelImage, modelKeyPoints, observedImage, observedKeyPoints,
               indices, new Bgr(255, 255, 255), new Bgr(255, 255, 255), mask, Features2DToolbox.KeypointDrawType.DEFAULT);

            #region draw the projected region on the image
            if (homography != null)
            {  //draw a rectangle along the projected model
                Rectangle rect = modelImage.ROI;
                PointF[] pts = new PointF[] {
         new PointF(rect.Left, rect.Bottom),
         new PointF(rect.Right, rect.Bottom),
         new PointF(rect.Right, rect.Top),
         new PointF(rect.Left, rect.Top)};
                homography.ProjectPoints(pts);

                result.DrawPolyline(Array.ConvertAll<PointF, Point>(pts, Point.Round), true, new Bgr(Color.Red), 5);
            }
            #endregion

            return result;
        }
        */

    }
}
