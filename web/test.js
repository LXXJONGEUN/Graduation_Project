const {PythonShell} = require('python-shell');

let options = {
    mode: 'text',
    pythonPath: '',
    pythonOptions: ['-u'], // get print results in real-time
    scriptPath: '',
    args: ['value1', 'value2', 'value3']
};

PythonShell.run('test.py', options, function(err, results){
    if (err) {
        console.log('1')
        throw err;
    
    }
    console.log('finished');
    console.log(results);
});