var express = require('express');
var router = express.Router();

/* GET home page. */
router.get('/', function(req, res, next) {
  res.render('index', { title: 'Express' });
});

module.exports = router;


router.post('/result', function(req, res, next){
	var data_set = req.body.data_set;
	var algorithm = req.body.algorithm;
	var model = req.body.model;
	var test_set = req.body.test_set;
	var mnist_flag = false
	if (data_set == "MNIST"){
		console.log(data_set)
		mnist_flag = true
	}
	var predict

	/////

	const {PythonShell} = require('python-shell');

	let options = {
		mode: 'text',
		pythonPath: '',
		pythonOptions: ['-u'], // get print results in real-time
		scriptPath: '',
		args: ['value1', 'value2', 'value3']
	};

	PythonShell.run(data_set + '_' + algorithm + '_' + model + '_test.py', options, function(err, results){
		if (err) {
			console.log('1')
			throw err;
		
		}
		console.log('finished');
		console.log(results);
		predict = results
		flag = true
		res.render('result', {data_set : data_set, algorithm : algorithm, model : model, predict : predict, mnist_flag : mnist_flag})
	})	
})

router.get('/result', function(req, res, next){
	res.send('Wrong Access');
})