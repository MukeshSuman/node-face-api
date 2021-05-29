var express = require('express');
var router = express.Router();
var path = require('path');

// import nodejs bindings to native tensorflow,
// not required, but will speed up things drastically (python required)
// require('@tensorflow/tfjs-node');

// implements nodejs wrappers for HTMLCanvasElement, HTMLImageElement, ImageData
var canvas = require('canvas');
var faceapi = require('face-api.js');


// patch nodejs environment, we need to provide an implementation of
// HTMLCanvasElement and HTMLImageElement
const { Canvas, Image, ImageData } = canvas
faceapi.env.monkeyPatch({ Canvas, Image, ImageData })


const faceDetectionNet = faceapi.nets.ssdMobilenetv1

// SsdMobilenetv1Options
const minConfidence = 0.5

// TinyFaceDetectorOptions
const inputSize = 416
const scoreThreshold = 0.5

// MtcnnOptions
const minFaceSize = 50
const scaleFactor = 0.8

function getFaceDetectorOptions(net) {
  return net === faceapi.nets.ssdMobilenetv1
    ? new faceapi.SsdMobilenetv1Options({ minConfidence })
    : (net === faceapi.nets.tinyFaceDetector
      ? new faceapi.TinyFaceDetectorOptions({ inputSize, scoreThreshold })
      : new faceapi.MtcnnOptions({ minFaceSize, scaleFactor })
    )
}

const faceDetectionOptions = getFaceDetectorOptions(faceDetectionNet)

const setup = async () => {
  // load weights
  await faceDetectionNet.loadFromDisk('weights')
  // await faceapi.nets.faceLandmark68Net.loadFromDisk('weights')
  // await faceapi.loadFaceLandmarkModel('/')
  await faceapi.nets.ageGenderNet.loadFromDisk('weights')

}

const testImage = async () => {
  console.time("testImage");
  await setup()
  // load the image
  const img = await canvas.loadImage('imgs_src/da.jpeg')

  const detectionWithAgeAndGender = await faceapi.detectSingleFace(img, faceDetectionOptions).withAgeAndGender()
  console.log("detectionWithAgeAndGender", detectionWithAgeAndGender)
  console.timeEnd("testImage")

}

const testImage1 = async () => {
  console.time("dbsave");
  // load the image
  const img = await canvas.loadImage('imgs_src/da.jpeg')

  const detectionWithAgeAndGender = await faceapi.detectSingleFace(img).withAgeAndGender()
  console.log("detectionWithAgeAndGender", detectionWithAgeAndGender)
  console.timeEnd("dbsave")
}



/* GET users listing. */
router.get('/', function (req, res, next) {
  testImage()
  res.writeHead(200, { 'Content-Type': 'text/html' });
  res.end('<head></head><body><div>test</div></body>');
  // res.send();
});

/* GET users listing. */
router.get('/test', function (req, res, next) {
  testImage1()
  res.writeHead(200, { 'Content-Type': 'text/html' });
  res.end('<head></head><body><div>testImage1</div></body>');
  // res.send();
});

module.exports = router;
