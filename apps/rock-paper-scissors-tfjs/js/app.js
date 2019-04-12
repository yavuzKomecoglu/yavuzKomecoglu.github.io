//------------------
// GLOBAL VARIABLES
//------------------
var isMobileOrTablet = true;
var isWebcamOn = 0;
var modelURL = "https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json";
var NUM_CLASSES = 3;
var predictCount = 0;
var predictMax = 1000;

var imgWidth = 400;
var imgHeight = 400;

let extractor;
let classifier;
let xs;
let ys;

var SAMPLE_BOX = {
	0: 0,
	1: 0,
	2: 0
}

var CLASS_MAP = {
	0: "emoticon-rock",
	1: "emoticon-paper",
	2: "emoticon-scissors"
}

Webcam.set({
	width: imgWidth,
	height: imgHeight,
	image_format: 'jpeg',
	jpeg_quality: 100,
	crop_width: imgWidth,
	crop_height: imgHeight
});


var gestureComputerIds = [
	'rock-pc',
	'paper-pc',
	'scissors-pc',
];

var winnerMatrix = [
	[0, 1, -1],
	[-1, 0, 1],
	[1, -1, 0],
];


var gaming = false;
var currentMove = -1;


//-----------------------------
// disable support for mobile 
// and tablet
//-----------------------------
function mobileAndTabletcheck() {
	var check = false;
	(function (a) {
		//
	})(navigator.userAgent || navigator.vendor || window.opera);
	isMobileOrTablet = check;
};

window.onload = function () {
	mobileAndTabletcheck();
	if (isMobileOrTablet) {
		document.getElementById("game-container").style.display = "none";
		document.getElementById("mobile-tablet-warning").style.display = "block";
	} else {
		startWebcam();
		loadExtractor();
	}
}

//-----------------------
// start webcam capture
//-----------------------
function startWebcam() {
	predictCount = 0;
	var video = document.getElementById('main-stream-video');

	Webcam.attach('#main-stream-video');

	Webcam.on('live', function () {
		isWebcamOn = 1;
	});

	Webcam.on('error', function (err) {
		console.log(err);
		console.log("Web kamerasına erişim sağlanamıyor!");
		isWebcamOn = 0;
	});

}

//---------------------
// stop webcam capture
//---------------------
function stopWebcam() {
	/*var video = document.getElementById('main-stream-video');*/
	/*video.stop();*/
	Webcam.reset();
	isWebcamOn = 0;
	predictCount = predictMax + 1;
}

//------------------------------
// capture webcam stream and 
// assign it to a canvas object
//------------------------------
function captureWebcam() {

	var MyCanvas = document.createElement("canvas");
	var MyContext = MyCanvas.getContext('2d');
	MyCanvas.width = imgWidth;
	MyCanvas.height = imgHeight;

	Webcam.snap(function (data_uri, canvas, context) {

		MyContext.drawImage(canvas, 0, 0);
		//console.log(MyContext.canvas);
		tensor_image = preprocessImage(MyCanvas);

	});

	var canvasObj = {
		canvasElement: MyCanvas,
		canvasTensor: tensor_image
	};

	return canvasObj;

}

//---------------------------------
// take snapshot for each category
//---------------------------------
function captureSample(id, label) {
	if (isWebcamOn == 1) {
		canvasObj = captureWebcam();

		canvas = canvasObj["canvasElement"];
		tensor_image = canvasObj["canvasTensor"];

		var img_id = id.replace("sample", "image");
		var img = document.getElementById(img_id);
		img.src = canvas.toDataURL();

		// add the sample to the training tensor
		addSampleToTensor(extractor.predict(tensor_image), label);

		SAMPLE_BOX[label] += 1;
		document.getElementById(id.replace("sample", "count")).innerHTML = SAMPLE_BOX[label] + " örnek";




	} else {
		alert("Lütfen önce 'Başlat' buttonu ile web kamerasını açın!")
	}
}

//------------------------------------
// preprocess the image from webcam
// to be mobilenet friendly
//------------------------------------
function preprocessImage(img) {
	const tensor = tf.browser.fromPixels(img).resizeNearestNeighbor([224, 224]);
	const croppedTensor = cropImage(tensor);
	const batchedTensor = croppedTensor.expandDims(0);

	return batchedTensor.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
}

//------------------------------------
// crop the image from the webcam
// region of interest: center portion
//------------------------------------
function cropImage(img) {
	const size = Math.min(img.shape[0], img.shape[1]);
	const centerHeight = img.shape[0] / 2;
	const beginHeight = centerHeight - (size / 2);
	const centerWidth = img.shape[1] / 2;
	const beginWidth = centerWidth - (size / 2);
	return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}

//------------------------------------
// hold each sample as a tensor that
// has 4 dimensions
// https://js.tensorflow.org/api/latest/#oneHot
//------------------------------------
function addSampleToTensor(sample, label) {
	const y = tf.tidy(
		() => tf.oneHot(tf.tensor1d([label]).toInt(), NUM_CLASSES));

	//console.log()
	console.log("label", label)
	tf.oneHot(tf.tensor1d([label]).toInt(), NUM_CLASSES).print();


	if (xs == null) {
		xs = tf.keep(sample);
		ys = tf.keep(y);
	} else {
		const oldX = xs;
		xs = tf.keep(oldX.concat(sample, 0));
		const oldY = ys;
		ys = tf.keep(oldY.concat(y, 0));
		oldX.dispose();
		oldY.dispose();
		y.dispose();
	}
}

//------------------------------------
// train the classifier with the 
// obtained tensors from the user
//------------------------------------
async function train() {
	var selectLearningRate = document.getElementById("game-learning-rate");
	const learningRate = selectLearningRate.options[selectLearningRate.selectedIndex].value;

	var selectBatchSize = document.getElementById("game-batch-size");
	const batchSizeFrac = selectBatchSize.options[selectBatchSize.selectedIndex].value;

	var selectEpochs = document.getElementById("game-epochs");
	const epochs = selectEpochs.options[selectEpochs.selectedIndex].value;

	var selectHiddenUnits = document.getElementById("game-hidden-units");
	const hiddenUnits = selectHiddenUnits.options[selectHiddenUnits.selectedIndex].value;

	if (xs == null) {
		alert("Eğitime başlatmadan önce sınıflar için örnek resimleri çekin!");
	} else {
		classifier = tf.sequential({
			layers: [
				tf.layers.flatten({
					inputShape: [7, 7, 256]
				}),
				tf.layers.dense({
					units: parseInt(hiddenUnits),
					activation: "relu",
					kernelInitializer: "varianceScaling",
					useBias: true
				}),
				tf.layers.dense({
					units: parseInt(NUM_CLASSES),
					kernelInitializer: "varianceScaling",
					useBias: false,
					activation: "softmax"
				})
			]
		});
		const optimizer = tf.train.adam(learningRate);
		classifier.compile({
			optimizer: optimizer,
			loss: "categoricalCrossentropy"
		});

		const batchSize = Math.floor(xs.shape[0] * parseFloat(batchSizeFrac));
		if (!(batchSize > 0)) {
			alert("Lütfen küme boyutunu (batch size) sıfırdan büyük seçin!")
		}

		// create loss visualization
		var lossTextEle = document.getElementById("game-loss");
		if (typeof (lossTextEle) != 'undefined' && lossTextEle != null) {
			lossTextEle.innerHTML = "";
		} else {
			var lossText = document.createElement("P");
			lossText.setAttribute("id", "game-loss");
			lossText.classList.add('game-loss');
			document.getElementById("game-controller").insertBefore(lossText, document.getElementById("game-controller").children[1]);
			var lossTextEle = document.getElementById("game-loss");
		}

		classifier.fit(xs, ys, {
			batchSize,
			epochs: parseInt(epochs),
			callbacks: {
				onBatchEnd: async (batch, logs) => {
					lossTextEle.innerHTML = "Loss(yitim): " + logs.loss.toFixed(5);
					await tf.nextFrame();
				}
			}
		});
	}
}


//-------------------------------------
// load mobilenet model from Google
// and return a model that has the
// internal activations from a 
// specific feature layer in mobilenet
//-------------------------------------
async function loadExtractor() {
	// load mobilenet from Google
	const mobilenet = await tf.loadLayersModel(modelURL);


	// return the mobilenet model with 
	// internal activations from "conv_pw_13_relu" layer
	const feature_layer = mobilenet.getLayer("conv_pw_13_relu");

	// return mobilenet model with feature activations from specific layer
	extractor = tf.model({
		inputs: mobilenet.inputs,
		outputs: feature_layer.output
	});
}

//------------------------------
// Predict what the user plays
//------------------------------
var isPredicting = false;
async function predictPlay() {
	isPredicting = true;

	while (isPredicting) {
		const predictedClass = tf.tidy(() => {
			canvasObj = captureWebcam();
			canvas = canvasObj["canvasElement"];
			const img = canvasObj["canvasTensor"];
			const features = extractor.predict(img);
			const predictions = classifier.predict(features);
			return predictions.as1D().argMax();
		});

		const classId = (await predictedClass.data())[0];
		predictedClass.dispose();
		highlightTile(classId);

		await tf.nextFrame();
	}
}

//------------------------------------------
// highlight the emoticon corresponding to
// user's game
//------------------------------------------
function highlightTile(classId) {
	currentMove = classId;
	console.log("currentMove",currentMove);

	var tile_play = document.getElementById(CLASS_MAP[classId].replace("emoticon", "game"));

	var tile_plays = document.getElementsByClassName("game-kit-comps");
	for (var i = 0; i < tile_plays.length; i++) {
		tile_plays[i].style.borderColor = "#e9e9e9";
		tile_plays[i].style.backgroundColor = "#ffffff";
		tile_plays[i].style.transform = "scale(1.0)";
	}

	tile_play.style.borderColor = "#ffffff";
	tile_play.style.backgroundColor = "#ffcc4d";
	tile_play.style.transform = "scale(1.2)";
}

var myTimer = 0;

function startTimer(timer, display) {
	console.log("timer",timer);
	myTimer = setInterval(function () {
		if (gaming) {
			var minutes = parseInt(timer / 60, 10)
			var seconds = parseInt(timer % 60, 10);

			console.log("minutes",minutes);
			console.log("seconds",seconds);

			minutes = minutes < 10 ? "0" + minutes : minutes;
			seconds = seconds < 10 ? "0" + seconds : seconds;

			display.textContent = seconds + " saniye kaldı. Hamleni hazırla!";


			var computerMove = Math.floor(Math.random() * 3);
			console.log("computerMove", computerMove);

			if (timer <= 0) {
				clearInterval(myTimer); 
				resolveGame();
				
			}

			--timer;
		}

	}, 1000);


}

function startGame() {
	if (xs == null) {
		alert("Oyunu başlatmadan önce sınıflar için örnek resimleri çekin! Ve eğitin!");
	}
	else
	{
		var startButton = document.getElementById('btn-start-game');

		if (startButton.disabled) {
			return;
		}
		gaming = true;
		startButton.disabled = true;
	
		var img = document.getElementById("image-human");
		img.src = "";
	
		for (let i = 0; i < 3; i++) {
			var gestureComputerElement = document.getElementById(gestureComputerIds[i]);
	
			var isHide = gestureComputerElement.classList.contains("hide");
			if(!isHide)
			{
				gestureComputerElement.classList.add("hide");
			}
			
		}
	
	
		var timerSeconds = 5;
		var gameStatus = document.getElementById('game-status');
	
		startTimer(timerSeconds, gameStatus);
	
	}

}

//------------------------------
// Resolve the game
//------------------------------
function resolveGame() {
	console.log("resolveGame");

	var gameStatus = document.getElementById('game-status');

	gaming = false;
	isPredicting=false;
	var computerMove = Math.floor(Math.random() * 3);
	var result = winnerMatrix[computerMove][currentMove];
	console.log("result",result)
	switch (result) {
		case -1:
			gameStatus.textContent = 'Kaybettin. Tekrar deneyin!';
			break;
		case 0:
			gameStatus.textContent = 'Berabere! Tekrar deneyin.';
			break;
		case 1:
			gameStatus.textContent = 'Sen kazandın. Tebrikler! :)'
			break;
	}

	//var computerMoveElement = document.getElementById(gestureComputerIds[computerMove]);
	//console.log("computerMoveElement",computerMoveElement);

	//computerMoveElement.classList.remove("hide");
	
	
	for (let i = 0; i < 3; i++) {
		var gestureComputerElement = document.getElementById(gestureComputerIds[i]);

		if(i == computerMove)
		{
			gestureComputerElement.classList.remove("hide");
		}
		
	}


	for (let i = 0; i < 3; i++) {
		//this.gestureCpuImages[i].hidden = (i !== computerMove);
	}
	var startButton = document.getElementById('btn-start-game');
	startButton.disabled = false;


	canvasObj = captureWebcam();

	canvas = canvasObj["canvasElement"];
	tensor_image = canvasObj["canvasTensor"];

	var img = document.getElementById("image-human");
	img.src = canvas.toDataURL();


}