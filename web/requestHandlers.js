// var path=require("path");
// var jsdom = require("jsdom");
// var multiparty = require('multiparty');
var querystring = require("querystring"),
    fs = require("fs"),
    formidable = require("formidable"),
    execSync = require("child_process").execSync;

function start(response) {
    console.log("Request handler 'start' was called.");

    var body = '<html style="background-color: antiquewhite">' +
        '<head>' +
        '<meta http-equiv="Content-Type" content="text/css; ' +
        'charset=UTF-8" />' +
        '<title>' +
        'BrainImageNet: Sex Prediction</title>' +
        '</head>' +
        '<body marginwidth="300px" bgcolor="#faebd7" marginheight="100px" style="background-color: lightsteelblue; width: 600px; padding: 0 60px 60px 60px; border: 2px solid greenyellow">' +
        '<h1 style="font-size: 48px;\n' +
        ' text-align: center;\n' +
        ' margin: 0;\n' +
        ' padding: 20px 0;\n' +
        ' color: palegoldenrod;\n' +
        ' text-shadow: 3px 3px 1px ghostwhite;">Welcome!</h1>' +
        '<img src="logo" alt="DPABI" height="500px" width="=500px" align="center">' +
        // '<div style="color:#00FF00">' +
        // '<h3>This is a header</h3>' +
        // '<p>This is a paragraph.</p>' +
        // '</div>' +
        // '<div class="news">' +
        // '<h2>News headline 1</h2>' +
        // '<p>some text. some text. some text...</p>' +
        // '</div>' +
        // '<div class="news">' +
        // '<h2>News headline 2</h2>' +
        // '<p>some text. some text. some text...</p>' +
        // '</div>' +
        // '<div class="hp_sw_logo hpcLogoWhite"><h1 class="a11yhide">必应</h1>必应</div>'+
        // '<div id="est_switch"><div id="est_cn" class="est_selected">国内版</div><div id="est_en" class="est_unselected">国际版</div>'+
        // '<div class="box" style="animation: alternate">'+
        // '<ul style="animation: alternate; box-shadow: coral">'+
        // '<li>'+
        // '<input  type="button" name="check" id="active1" checked onclick="start(response)">' +
        '<br>' +
        '<button onclick="window.location.href=\'/sex\'" style="background-color: darkorange">Sex Prediction</button>'+
        // '<div>111</div>'+
        // '</li>'+
        // '<li>'+
        // '<input type="radio" name="check" id="active2">' +
        '<button onclick="window.location.href=\'/disease\'" style="background-color: burlywood">AD Prediction</button>'+
        // '<div>222</div>'+
        // '</li>'+
        // '</ul>'+
        // '</div>'+
        '<br>' +
        // '<a href="/sex">Sex Prediction</a>' +
        // '<a href="/disease">Disease Prediction</a>' +
        '<form action="/email" enctype="multipart/form-data" ' +
        'method="post">' +
        '<p style="font-size: 16px;\n' +
        'line-height: 2;\n' +
        'letter-spacing: 1px;\n' +
        'font-family: Arial;">This is a website to demonstrate <strong>BrainImageNet: an Industrial-Grade Brain Imaging Based Deep Learning Classifier</strong>. We openly shared our trained model, code and framework (<a href="https://github.com/Chaogan-Yan/BrainImageNet"><strong>https://github.com/Chaogan-Yan/BrainImageNet</strong></a>), and here built an online predicting website for whoever are interested in testing our classifier with brain imaging data from anybody and any scanner. Please upload brain structural data (raw data or preprocessed gray matter density/volume data) to predict the sex of the participant(s). OF NOTE: FOR RESEARCH PURPOSE ONLY!\n</p>' +
        '<br>' +
        '<p>1. Predicting sex by preprocessed gray matter density/volume data. This will significantly enhance the speed of prediction. Please preprocess your structural image to get the gray matter density data (wc1*) and gray matter volume data (mwc1*) by SPM or DPABI/DPARSF. The data required is NIfTI format (.nii or .nii.gz) in MNI space with a resolution of 91*109*91. The filenames should be wc1_XXXXX.nii(.gz) and mwc1_XXXXX.nii(.gz)  (XXXXX is subject ID). Please compress all of your files (could be many participants) in a single .zip file for uploading. Please see an example of <a href="https://github.com/Chaogan-Yan/BrainImageNet/blob/master/data/DemoData_BrainImageNet.zip"><strong>DemoData_BrainImageNet</strong></a>. As the prediction may take minutes (even tens of minutes depending on the number of participants), you can wait here or leave your email in the textbox to receive the prediction results from email attachment.</p>' +
        '<br>' +
        '<input type="file" id="uploadfile" name="upload" multiple="multiple" accept="application/zip, application/gzip">' +
        '<input type="email" id="email1" name="emailaddress" multiple="multiple">' +
        '<input type="submit" value="Predict by wc1&mwc1" multiple="multiple"/>' +
        '</form>' +
        '<form action="/email2" enctype="multipart/form-data" ' +
        'method="post">' +
        '<br>' +
        '<p>2. Predicting sex by raw T1 structural image. The data required is NIfTI format (.nii or .nii.gz). Please compress all of your files (could be many participants) in a single .zip file for uploading. As the prediction may take tens of minutes depending on the number of participants, you can wait here or leave your email in the textbox to receive the prediction results from email attachment.</p>' +
        '<input type="file" id="upload" name="upload" multiple="multiple" accept="application/zip, application/gzip">' +
        '<input type="email" id="email2" name="emailaddress" multiple="multiple">' +
        '<input type="submit" value="Predict by T1" multiple="multiple"/>' +
        '</form>' +
        '<br>' +
        '<p style="line-height: 2">For more information about our research, please visit <a href="http://yanlab.psych.ac.cn/">The R-fMRI Lab</a>.</p>' +
        '</body>' +
        '</html>';
    response.writeHead(200, {"Content-Type": "text/html"});
    response.write(body);
    response.end();
}

function disease(response) {
    console.log("Request handler 'disease' was called.");

    var body = '<html style="background-color: antiquewhite">' +
        '<head>' +
        '<meta http-equiv="Content-Type" content="text/css; ' +
        'charset=UTF-8" />' +
        '<title>' +
        'BrainImageNet: AD Prediction</title>' +
        '</head>' +
        '<body marginwidth="300px" bgcolor="#faebd7" marginheight="100px" style="background-color: lightsteelblue; width: 600px; padding: 0 60px 60px 60px; border: 2px solid greenyellow">' +
        '<h1 style="font-size: 48px;\n' +
        ' text-align: center;\n' +
        ' margin: 0;\n' +
        ' padding: 20px 0;\n' +
        ' color: palegoldenrod;\n' +
        ' text-shadow: 3px 3px 1px ghostwhite;">Welcome!</h1>' +
        '<img src="logo" alt="DPABI" height="500px" width="=500px" align="center">' +
        '<br>' +
        '<br>' +
        '<button onclick="window.location.href=\'/sex\'" style="background-color: burlywood">Sex Prediction</button>'+
        '<button onclick="window.location.href=\'/disease\'" style="background-color: darkorange">AD Prediction</button>'+
        '<br>' +
        // '<a href="/sex">Sex Prediction</a>' +
        // '<a href="/disease">Disease Prediction</a>' +
        '<form action="/email3" enctype="multipart/form-data" ' +
        'method="post">' +
        '<p style="font-size: 16px;\n' +
        'line-height: 2;\n' +
        'letter-spacing: 1px;\n' +
        'font-family: Arial;">This is a website to demonstrate <strong>BrainImageNet: an Industrial-Grade Brain Imaging Based Deep Learning Classifier</strong>. We openly shared our trained model, code and framework (<a href="https://github.com/Chaogan-Yan/BrainImageNet"><strong>https://github.com/Chaogan-Yan/BrainImageNet</strong></a>), and here built an online predicting website for whoever are interested in testing our classifier with brain imaging data from anybody and any scanner. Please upload brain structural data (raw data or preprocessed gray matter density/volume data) to predict the Alzheimers Disease (AD) status of the participant(s). OF NOTE: FOR RESEARCH PURPOSE ONLY!\n</p>' +
        '<br>' +
        '<p>1. Predicting Alzheimers Disease (AD) by preprocessed gray matter density/volume data. This will significantly enhance the speed of prediction. Please preprocess your structural image to get the gray matter density data (wc1*) and gray matter volume data (mwc1*) by SPM or DPABI/DPARSF. The data required is NIfTI format (.nii or .nii.gz) in MNI space with a resolution of 91*109*91. The filenames should be wc1_XXXXX.nii(.gz) and mwc1_XXXXX.nii(.gz)  (XXXXX is subject ID). Please compress all of your files (could be many participants) in a single .zip file for uploading. Please compress all of your files (could be many participants) in a single .zip file for uploading. Please see an example of <a href="https://github.com/Chaogan-Yan/BrainImageNet/blob/master/data/DemoData_BrainImageNet.zip"><strong>DemoData_BrainImageNet</strong></a>. As the prediction may take minutes (even tens of minutes depending on the number of participants), you can wait here or leave your email in the textbox to receive the prediction results from email attachment.</p>' +
        '<br>' +
        '<input type="file" id="uploadfile" name="upload" multiple="multiple" accept="application/zip, application/gzip">' +
        '<input type="email" id="email1" name="emailaddress" multiple="multiple">' +
        '<input type="submit" value="Predict by wc1&mwc1" multiple="multiple"/>' +
        '</form>' +
        '<form action="/email4" enctype="multipart/form-data" ' +
        'method="post">' +
        '<br>' +
        '<p>2. Predicting Alzheimers Disease (AD) by raw T1 structural image. The data required is NIfTI format (.nii or .nii.gz). Please compress all of your files (could be many participants) in a single .zip file for uploading. As the prediction may take minutes (even tens of minutes depending on the number of participants), you can wait here or leave your email in the textbox to receive the prediction results from email attachment.</p>' +
        '<input type="file" id="upload" name="upload" multiple="multiple" accept="application/zip, application/gzip">' +
        '<input type="email" id="email2" name="emailaddress" multiple="multiple">' +
        '<input type="submit" value="Predict by T1" multiple="multiple"/>' +
        '</form>' +
        '<br>' +
        '<p style="line-height: 2">For more information about our research, please visit <a href="http://yanlab.psych.ac.cn/">The R-fMRI Lab</a>.</p>' +
        '</body>' +
        '</html>';
    response.writeHead(200, {"Content-Type": "text/html"});
    response.write(body);
    response.end();
}

function upload(response, request) {
    console.log("Request handler 'upload' was called.");

    var form = new formidable.IncomingForm();
    // var form2=  new multiparty.Form({uploadDir: '../static/images/'});
    //
    // form2.parse(request, function(err, fields, files) {
    //     var filesTmp = JSON.stringify(files,null,2);
    //     console.log(files);
    //     if(err){
    //     } else {
    //         var inputFile = files.file[0];
    //         console.log(inputFile);
    //         var uploadedPath = inputFile.path;
    //         var dstPath = '../static/images/' + inputFile.originalFilename;
    //         fs.renameSync(uploadedPath, dstPath);
    //     }
    //     // res.send({"fileName":inputFile.originalFilename});
    //
    // });
    console.log("about to parse");
    form.parse(request, function (error, fields, files) {
        console.log("parsing done");
        try {
            let SubfolderName = getCurrentDate(2);
            response.writeHead(200, {"Content-Type": "text/html"});
            response.write("The Predictions:<br/>");
            fs.mkdirSync("DataUpload/" + SubfolderName);
            fs.mkdirSync("DataUpload/" + SubfolderName + "/UNZIP/");
            fs.mkdirSync("Predictions/" + SubfolderName);
            console.log("Upload:");
            let newpath = "/tmp/" + files.upload.name;
            let finalpath = "./DataUpload/" + SubfolderName + "/" + files.upload.name;
            console.log("to the new path:" + newpath);
            fs.renameSync(files.upload.path, newpath);
            console.log("upload finished");
            console.log("to the final path:" + finalpath);
            fs.copyFileSync(newpath, finalpath);
            console.log("copy finished");
            // console.log(finalpath.substring(finalpath.length-3,finalpath.length-1));
            if (finalpath.substring(finalpath.length - 3, finalpath.length) === "zip") {
                var cmdStr = "unzip " + finalpath + " -d " + "./DataUpload/" + SubfolderName + "/UNZIP/";

            } else if (finalpath.substring(finalpath.length - 2, finalpath.length) === "gz") {
                var cmdStr = "gzip -d " + finalpath;
            } else {
                var cmdStr = "unzip " + finalpath + " -d " + "./DataUpload/" + SubfolderName + "/UNZIP/";
            }
            execSync(cmdStr);
            // cmdStr2 = "python -c \"from Brain_Prediction import Gender_Classifier; Gender_Classifier('./DataUpload/"+SubfolderName+"/UNZIP','./Predictions/"+SubfolderName+"','aaa');\"";
            // execSync(cmdStr2);
            // dataf = fs.readFileSync("./Predictions/"+SubfolderName+"/Prediction.txt", "utf-8");
            //
            // response.write("<img src='/logo' height=\"500px\" width=\"=500px\" align=\"center\"/>");
            // response.write("<br/>");
            // response.write("<table border='1'>");
            // response.write("<tr><th>Subject_ID</th><th>Prediction</th></tr>");
            // response.write("<tr><td>");
            // for(var i=0;i<dataf.length;i++){
            //     // if(dataf[i]===' '){
            //     //     response.write("</td>");
            //     //     response.write("<td>");
            //     // } else{
            //     //
            //     //     response.write(dataf[i]);
            //     //
            //     //
            //     // }
            //     switch (dataf[i]) {
            //         case ' ' :
            //             response.write("</td>");
            //             response.write("</tr>");
            //             if(i<dataf.length-2){
            //                 response.write("<tr>");
            //                 response.write("<td>");
            //             }
            //             break;
            //         case '\\' :
            //             response.write("</td>");
            //             response.write("<td>");
            //             break;
            //         default :
            //             if(i<dataf.length-2){
            //                 response.write(dataf[i]);
            //             }
            //
            //
            //     }
            // }
            // response.write("</table>");
            // response.write("<br/>");
            // response.write("Prediction Finished. If there is no result, please check the form of your input.");
            // response.write("<br/>");
            // response.write("1 <- Male   0 <- Female");
            response.end();
        }
        catch {
            response.writeHead(200, {"Content-Type": "text/html"});
            response.write("Something wrong.\n");
            response.write("Please check your files.");
            response.end();
        }
    });
}

function upload2(response, request) {
    console.log("Request handler 'upload2' was called.");

    var form = new formidable.IncomingForm();
    console.log("about to parse");
    form.parse(request, function (error, fields, files) {
        console.log("parsing done");
        try {
            let SubfolderName = getCurrentDate(2);
            fs.mkdirSync("DataUpload/" + SubfolderName);
            fs.mkdirSync("DataUpload/" + SubfolderName + "/UNZIP/");
            fs.mkdirSync("DataUpload/" + SubfolderName + "/wc1/");
            fs.mkdirSync("Predictions/" + SubfolderName);
            console.log("Upload:");
            let newpath = "/tmp/" + files.upload.name;
            let finalpath = "./DataUpload/" + SubfolderName + "/" + files.upload.name;
            console.log("to the new path:" + newpath);
            fs.renameSync(files.upload.path, newpath);
            console.log("upload finished");
            console.log("to the final path:" + finalpath);
            fs.copyFileSync(newpath, finalpath);
            console.log("copy finished");
            if (finalpath.substring(finalpath.length - 3, finalpath.length) === "zip") {
                var cmdStr = "unzip " + finalpath + " -d " + "./DataUpload/" + SubfolderName + "/UNZIP/";

            } else if (finalpath.substring(finalpath.length - 2, finalpath.length) === "gz") {
                var cmdStr = "gzip -d " + finalpath;
            } else {
                var cmdStr = "unzip " + finalpath + " -d " + "./DataUpload/" + SubfolderName + "/UNZIP/";
            }
            execSync(cmdStr);
            cmdStr0 = "python -c \"import os; from generate_subid import generates; generates(os.path.abspath('./DataUpload/" + SubfolderName + "/UNZIP'));\"";
            console.log(cmdStr0);
            execSync(cmdStr0);
            console.log("subid write finished");
            cmdStr1 = "sh run_y_Segment.sh /opt/mcr/v95 ./DataUpload/" + SubfolderName + "/UNZIP/subid.txt";
            console.log(cmdStr1);
            execSync(cmdStr1);
            cmdStr2 = "python -c \"from transdoc import trans; trans('./DataUpload/" + SubfolderName + "/UNZIP','./DataUpload/" + SubfolderName + "/wc1');\"";
            execSync(cmdStr2);
            cmdStr3 = "python -c \"from Brain_Prediction import Gender_Classifier; Gender_Classifier('./DataUpload/" + SubfolderName + "/wc1','./Predictions/" + SubfolderName + "','aaa');\"";
            execSync(cmdStr3);
            dataf = fs.readFileSync("./Predictions/" + SubfolderName + "/Prediction.txt", "utf-8");
            response.writeHead(200, {"Content-Type": "text/html"});
            // response.write("Under Developing");
            response.write("The Predictions:<br/>");
            response.write("<img src='/logo' height=\"500px\" width=\"=500px\" align=\"center\"/>");
            response.write("<br/>");
            response.write("<table border='1'>");
            response.write("<tr><th>Subject_ID</th><th>Prediction</th></tr>");
            response.write("<tr><td>");
            for (var i = 0; i < dataf.length; i++) {
                switch (dataf[i]) {
                    case ' ' :
                        response.write("</td>");
                        response.write("</tr>");
                        if (i < dataf.length - 2) {
                            response.write("<tr>");
                            response.write("<td>");
                        }
                        break;
                    case '\\' :
                        response.write("</td>");
                        response.write("<td>");
                        break;
                    default :
                        if (i < dataf.length - 2) {
                            response.write(dataf[i]);
                        }


                }
            }
            response.write("</table>");
            response.write("<br/>");
            response.write("Prediction Finished. If there is no result, please check the form of your input.");
            response.write("<br/>");
            response.write("1: Male   0: Female. (Prediction close to 1 means male.)");
            response.end();
        }
        catch {
            response.writeHead(200, {"Content-Type": "text/html"});
            response.write("Something wrong.\n");
            response.write("Please check your files.");
            response.end();
        }
    });
}

function email(response, request) {
    console.log("Request handler 'email' was called.");

    var form = new formidable.IncomingForm();
    console.log("about to parse");
    var body = '<html style="background-color: antiquewhite">' +
        '<head>' +
        '<meta http-equiv="Content-Type" content="text/css; ' +
        'charset=UTF-8" />' +
        '<title>' +
        'BrainImageNet: Sex Prediction</title>' +
        '</head>' +
        '<body marginwidth="300px" bgcolor="#faebd7" marginheight="100px" style="background-color: lightsteelblue; width: 600px; padding: 0 60px 60px 60px; border: 2px solid greenyellow">' +
        '<h1 style="font-size: 48px;\n' +
        ' text-align: center;\n' +
        ' margin: 0;\n' +
        ' padding: 20px 0;\n' +
        ' color: darkorange;\n' +
        ' text-shadow: 3px 3px 1px ghostwhite;">BrainImageNet: \nSex Prediction</h1>' +
        '<img src="logo" alt="DPABI" height="500px" width="=500px" align="center">' +
        '<p style="line-height: 2">It will take some time (minutes or tens of minutes), please wait...</p>' +
        '</body>' +
        '</html>';
    response.writeHead(200, {"Content-Type": "text/html"});
    response.write(body);
    form.parse(request, function (error, fields, files) {
        wc1predict(response, error, fields, files);
    });

}

function wc1predict(response, error, fields, files) {
    let SubfolderName = getCurrentDate(2);
    cmdlog = "script log/" + SubfolderName + ".log";
    execSync(cmdlog);
    execSync("echo log begins");
    try {

        fs.mkdirSync("DataUpload/" + SubfolderName);
        fs.mkdirSync("DataUpload/" + SubfolderName + "/UNZIP/");
        fs.mkdirSync("Predictions/" + SubfolderName);
        console.log("Upload:");
        let newpath = "/tmp/" + files.upload.name;
        let finalpath = "./DataUpload/" + SubfolderName + "/" + files.upload.name;
        console.log("to the new path:" + newpath);
        fs.renameSync(files.upload.path, newpath);
        console.log("upload finished");
        console.log("to the final path:" + finalpath);
        fs.copyFileSync(newpath, finalpath);
        console.log("copy finished");
        if (finalpath.substring(finalpath.length - 3, finalpath.length) === "zip") {
            var cmdStr = "unzip " + finalpath + " -d " + "./DataUpload/" + SubfolderName + "/UNZIP/";

        } else if (finalpath.substring(finalpath.length - 2, finalpath.length) === "gz") {
            var cmdStr = "gzip -d " + finalpath;
        } else {
            var cmdStr = "unzip " + finalpath + " -d " + "./DataUpload/" + SubfolderName + "/UNZIP/";
        }
        execSync(cmdStr);
        cmdStr2 = "python -c \"from Brain_Prediction import Gender_Classifier; Gender_Classifier('./DataUpload/" + SubfolderName + "/UNZIP','./Predictions/" + SubfolderName + "','aaa');\"";
        execSync(cmdStr2);
        cmdStr3 = "python -c \"from sendemail import send; send('" + fields.emailaddress + "','./Predictions/" + SubfolderName + "/Prediction.txt');\"";
        if (fields.emailaddress !== '') {
            execSync(cmdStr3);
        }
        dataf = fs.readFileSync("./Predictions/" + SubfolderName + "/Prediction.txt", "utf-8");
        response.write("<br/>");
        response.write("<table border='1'>");
        response.write("<tr><th>Subject_ID</th><th>Prediction</th></tr>");
        response.write("<tr><td>");
        for (var i = 0; i < dataf.length; i++) {
            switch (dataf[i]) {
                case ' ' :
                    response.write("</td>");
                    response.write("</tr>");
                    if (i < dataf.length - 2) {
                        response.write("<tr>");
                        response.write("<td>");
                    }
                    break;
                case '\\' :
                    response.write("</td>");
                    response.write("<td>");
                    break;
                default :
                    if (i < dataf.length - 2) {
                        response.write(dataf[i]);
                    }


            }
        }
        response.write("</table>");
        response.write("<br/>");
        response.write("Prediction Finished. If there is no result, please check the form of your input.");
        response.write("<br/>");
        response.write("1: Male   0: Female. (Prediction close to 1 means male.)");
        cmdlog2 = "exit";
        execSync(cmdlog2);
        response.end();

    }
    catch {
        // response.writeHead(200, {"Content-Type": "text/html"});
        response.write("Something wrong.\n");
        response.write("Please check your files.");
        cmdlog2 = "exit";
        execSync(cmdlog2);
        response.end();
    }

}

function email2(response, request) {
    console.log("Request handler 'email2' was called.");

    var form = new formidable.IncomingForm();
    console.log("about to parse");
    var body = '<html style="background-color: antiquewhite">' +
        '<head>' +
        '<meta http-equiv="Content-Type" content="text/css; ' +
        'charset=UTF-8" />' +
        '<title>' +
        'BrainImageNet: Sex Prediction</title>' +
        '</head>' +
        '<body marginwidth="300px" bgcolor="#faebd7" marginheight="100px" style="background-color: lightsteelblue; width: 600px; padding: 0 60px 60px 60px; border: 2px solid greenyellow">' +
        '<h1 style="font-size: 48px;\n' +
        ' text-align: center;\n' +
        ' margin: 0;\n' +
        ' padding: 20px 0;\n' +
        ' color: darkorange;\n' +
        ' text-shadow: 3px 3px 1px ghostwhite;">BrainImageNet: \nSex Prediction</h1>' +
        '<img src="logo" alt="DPABI" height="500px" width="=500px" align="center">' +
        '<p style="line-height: 2">It will take some time (minutes or tens of minutes), please wait...</p>' +
        '</body>' +
        '</html>';
    response.writeHead(200, {"Content-Type": "text/html"});
    response.write(body);
    form.parse(request, function (error, fields, files) {
        console.log("parsing done");
        T1predict(request, response, error, fields, files);
    });
}

function T1predict(request, response, error, fields, files) {
    let SubfolderName = getCurrentDate(2);
    cmdlog = "script log/" + SubfolderName + ".log";
    execSync(cmdlog);
    try {

        fs.mkdirSync("DataUpload/" + SubfolderName);
        fs.mkdirSync("DataUpload/" + SubfolderName + "/UNZIP/");
        fs.mkdirSync("DataUpload/" + SubfolderName + "/wc1/");
        fs.mkdirSync("Predictions/" + SubfolderName);
        console.log("Upload:");
        let newpath = "/tmp/" + files.upload.name;
        let finalpath = "./DataUpload/" + SubfolderName + "/" + files.upload.name;
        console.log("to the new path:" + newpath);
        fs.renameSync(files.upload.path, newpath);
        console.log("upload finished");
        console.log("to the final path:" + finalpath);
        fs.copyFileSync(newpath, finalpath);
        console.log("copy finished");
        if (finalpath.substring(finalpath.length - 3, finalpath.length) === "zip") {
            var cmdStr = "unzip " + finalpath + " -d " + "./DataUpload/" + SubfolderName + "/UNZIP/";

        } else if (finalpath.substring(finalpath.length - 2, finalpath.length) === "gz") {
            var cmdStr = "cp " + finalpath + " ./DataUpload/" + SubfolderName + "/UNZIP/" + files.upload.name;
        } else {
            var cmdStr = "unzip " + finalpath + " -d " + "./DataUpload/" + SubfolderName + "/UNZIP/";
        }
        execSync(cmdStr);
        cmdStr0 = "python -c \"import os; from generate_subid import generates; generates(os.path.abspath('./DataUpload/" + SubfolderName + "/UNZIP'));\"";
        console.log(cmdStr0);
        execSync(cmdStr0);
        console.log("subid write finished");
        cmdStr1 = "sh run_y_Segment.sh /opt/mcr/v95 ./DataUpload/" + SubfolderName + "/UNZIP/subid.txt";
        console.log(cmdStr1);
        execSync(cmdStr1);
        cmdStr2 = "python -c \"from transdoc import trans; trans('./DataUpload/" + SubfolderName + "/UNZIP','./DataUpload/" + SubfolderName + "/wc1');\"";
        execSync(cmdStr2);
        cmdStr3 = "python -c \"from Brain_Prediction import Gender_Classifier; Gender_Classifier('./DataUpload/" + SubfolderName + "/wc1','./Predictions/" + SubfolderName + "','aaa');\"";
        execSync(cmdStr3);
        cmdStr4 = "python -c \"from sendemail import send; send('" + fields.emailaddress + "','./Predictions/" + SubfolderName + "/Prediction.txt');\"";
        if (fields.emailaddress !== '') {
            execSync(cmdStr4);
        }
        dataf = fs.readFileSync("./Predictions/" + SubfolderName + "/Prediction.txt", "utf-8");
        response.write("<br/>");
        response.write("<table border='1'>");
        response.write("<tr><th>Subject_ID</th><th>Prediction</th></tr>");
        response.write("<tr><td>");
        for (var i = 0; i < dataf.length; i++) {
            switch (dataf[i]) {
                case ' ' :
                    response.write("</td>");
                    response.write("</tr>");
                    if (i < dataf.length - 2) {
                        response.write("<tr>");
                        response.write("<td>");
                    }
                    break;
                case '\\' :
                    response.write("</td>");
                    response.write("<td>");
                    break;
                default :
                    if (i < dataf.length - 2) {
                        response.write(dataf[i]);
                    }
            }
        }
        response.write("</table>");
        response.write("<br/>");
        response.write("Prediction finished. If there is no result, please check the form of your input.");
        response.write("<br/>");
        response.write("1: Male   0: Female. (Prediction close to 1 means male.)");
        cmdlog2 = "exit";
        execSync(cmdlog2);
        response.end();
    }
    catch {
        response.write("Something wrong.\n");
        response.write("Please check your files.");
        cmdlog2 = "exit";
        execSync(cmdlog2);
        response.end();
    }
}


function email3(response, request) {
    console.log("Request handler 'email3' was called.");

    var form = new formidable.IncomingForm();
    console.log("about to parse");
    var body = '<html style="background-color: antiquewhite">' +
        '<head>' +
        '<meta http-equiv="Content-Type" content="text/css; ' +
        'charset=UTF-8" />' +
        '<title>' +
        'BrainImageNet: AD Prediction</title>' +
        '</head>' +
        '<body marginwidth="300px" bgcolor="#faebd7" marginheight="100px" style="background-color: lightsteelblue; width: 600px; padding: 0 60px 60px 60px; border: 2px solid greenyellow">' +
        '<h1 style="font-size: 48px;\n' +
        ' text-align: center;\n' +
        ' margin: 0;\n' +
        ' padding: 20px 0;\n' +
        ' color: darkorange;\n' +
        ' text-shadow: 3px 3px 1px ghostwhite;">BrainImageNet: \nAD Prediction</h1>' +
        '<img src="logo" alt="DPABI" height="500px" width="=500px" align="center">' +
        '<p style="line-height: 2">It will take some time (minutes or tens of minutes), please wait...</p>' +
        '</body>' +
        '</html>';
    response.writeHead(200, {"Content-Type": "text/html"});
    response.write(body);
    form.parse(request, function (error, fields, files) {
        wc1predict_AD(response, error, fields, files);
    });

}

function wc1predict_AD(response, error, fields, files) {
    let SubfolderName = getCurrentDate(2);
    cmdlog = "script log/" + SubfolderName + ".log";
    execSync(cmdlog);
    execSync("echo log begins");
    try {

        fs.mkdirSync("DataUpload/" + SubfolderName);
        fs.mkdirSync("DataUpload/" + SubfolderName + "/UNZIP/");
        fs.mkdirSync("Predictions/" + SubfolderName);
        console.log("Upload:");
        let newpath = "/tmp/" + files.upload.name;
        let finalpath = "./DataUpload/" + SubfolderName + "/" + files.upload.name;
        console.log("to the new path:" + newpath);
        fs.renameSync(files.upload.path, newpath);
        console.log("upload finished");
        console.log("to the final path:" + finalpath);
        fs.copyFileSync(newpath, finalpath);
        console.log("copy finished");
        if (finalpath.substring(finalpath.length - 3, finalpath.length) === "zip") {
            var cmdStr = "unzip " + finalpath + " -d " + "./DataUpload/" + SubfolderName + "/UNZIP/";

        } else if (finalpath.substring(finalpath.length - 2, finalpath.length) === "gz") {
            var cmdStr = "gzip -d " + finalpath;
        } else {
            var cmdStr = "unzip " + finalpath + " -d " + "./DataUpload/" + SubfolderName + "/UNZIP/";
        }
        execSync(cmdStr);
        cmdStr2 = "python -c \"from Brain_Prediction_AD import AD_Classifier; AD_Classifier('./DataUpload/" + SubfolderName + "/UNZIP','./Predictions/" + SubfolderName + "','aaa');\"";
        console.log(cmdStr2);
        execSync(cmdStr2);

        cmdStr3 = "python -c \"from sendemail import send; send('" + fields.emailaddress + "','./Predictions/" + SubfolderName + "/Prediction.txt');\"";
        if (fields.emailaddress !== '') {
            execSync(cmdStr3);
        }
        dataf = fs.readFileSync("./Predictions/" + SubfolderName + "/Prediction.txt", "utf-8");
        response.write("<br/>");
        response.write("<table border='1'>");
        response.write("<tr><th>Subject_ID</th><th>Prediction</th></tr>");
        response.write("<tr><td>");
        for (var i = 0; i < dataf.length; i++) {
            switch (dataf[i]) {
                case ' ' :
                    response.write("</td>");
                    response.write("</tr>");
                    if (i < dataf.length - 2) {
                        response.write("<tr>");
                        response.write("<td>");
                    }
                    break;
                case '\\' :
                    response.write("</td>");
                    response.write("<td>");
                    break;
                default :
                    if (i < dataf.length - 2) {
                        response.write(dataf[i]);
                    }


            }
        }
        response.write("</table>");
        response.write("<br/>");
        response.write("Prediction Finished. If there is no result, please check the form of your input.");
        response.write("<br/>");
        response.write("1: AD   0: Healthy. (Prediction close to 1 means AD.)");
        cmdlog2 = "exit";
        execSync(cmdlog2);
        response.end();

    }
    catch {
        response.write("Something wrong.\n");
        response.write("Please check your files.");
        cmdlog2 = "exit";
        execSync(cmdlog2);
        response.end();
    }

}

function email4(response, request) {
    console.log("Request handler 'email4' was called.");

    var form = new formidable.IncomingForm();
    console.log("about to parse");
    var body = '<html style="background-color: antiquewhite">' +
        '<head>' +
        '<meta http-equiv="Content-Type" content="text/css; ' +
        'charset=UTF-8" />' +
        '<title>' +
        'BrainImageNet: AD Prediction</title>' +
        '</head>' +
        '<body marginwidth="300px" bgcolor="#faebd7" marginheight="100px" style="background-color: lightsteelblue; width: 600px; padding: 0 60px 60px 60px; border: 2px solid greenyellow">' +
        '<h1 style="font-size: 48px;\n' +
        ' text-align: center;\n' +
        ' margin: 0;\n' +
        ' padding: 20px 0;\n' +
        ' color: darkorange;\n' +
        ' text-shadow: 3px 3px 1px ghostwhite;">BrainImageNet: \nAD Prediction</h1>' +
        '<img src="logo" alt="DPABI" height="500px" width="=500px" align="center">' +
        '<p style="line-height: 2">It will take some time (minutes or tens of minutes), please wait...</p>' +
        '</body>' +
        '</html>';
    response.writeHead(200, {"Content-Type": "text/html"});
    response.write(body);
    form.parse(request, function (error, fields, files) {
        console.log("parsing done");
        T1predict_AD(request, response, error, fields, files);
    });
}

function T1predict_AD(request, response, error, fields, files) {
    let SubfolderName = getCurrentDate(2);
    cmdlog = "script log/" + SubfolderName + ".log";
    execSync(cmdlog);
    try {

        fs.mkdirSync("DataUpload/" + SubfolderName);
        fs.mkdirSync("DataUpload/" + SubfolderName + "/UNZIP/");
        fs.mkdirSync("DataUpload/" + SubfolderName + "/wc1/");
        fs.mkdirSync("Predictions/" + SubfolderName);
        console.log("Upload:");
        let newpath = "/tmp/" + files.upload.name;
        let finalpath = "./DataUpload/" + SubfolderName + "/" + files.upload.name;
        console.log("to the new path:" + newpath);
        fs.renameSync(files.upload.path, newpath);
        console.log("upload finished");
        console.log("to the final path:" + finalpath);
        fs.copyFileSync(newpath, finalpath);
        console.log("copy finished");
        if (finalpath.substring(finalpath.length - 3, finalpath.length) === "zip") {
            var cmdStr = "unzip " + finalpath + " -d " + "./DataUpload/" + SubfolderName + "/UNZIP/";

        } else if (finalpath.substring(finalpath.length - 2, finalpath.length) === "gz") {
            var cmdStr = "cp " + finalpath + " ./DataUpload/" + SubfolderName + "/UNZIP/" + files.upload.name;
        } else {
            var cmdStr = "unzip " + finalpath + " -d " + "./DataUpload/" + SubfolderName + "/UNZIP/";
        }
        execSync(cmdStr);
        cmdStr0 = "python -c \"import os; from generate_subid import generates; generates(os.path.abspath('./DataUpload/" + SubfolderName + "/UNZIP'));\"";
        console.log(cmdStr0);
        execSync(cmdStr0);
        console.log("subid write finished");
        cmdStr1 = "sh run_y_Segment.sh /opt/mcr/v95 ./DataUpload/" + SubfolderName + "/UNZIP/subid.txt";
        console.log(cmdStr1);
        execSync(cmdStr1);
        cmdStr2 = "python -c \"from transdoc import trans; trans('./DataUpload/" + SubfolderName + "/UNZIP','./DataUpload/" + SubfolderName + "/wc1');\"";
        execSync(cmdStr2);
        cmdStr3 = "python -c \"from Brain_Prediction_AD import AD_Classifier; AD_Classifier('./DataUpload/" + SubfolderName + "/wc1','./Predictions/" + SubfolderName + "','aaa');\"";
        execSync(cmdStr3);
        cmdStr4 = "python -c \"from sendemail import send; send('" + fields.emailaddress + "','./Predictions/" + SubfolderName + "/Prediction.txt');\"";
        if (fields.emailaddress !== '') {
            execSync(cmdStr4);
        }
        dataf = fs.readFileSync("./Predictions/" + SubfolderName + "/Prediction.txt", "utf-8");
        response.write("<br/>");
        response.write("<table border='1'>");
        response.write("<tr><th>Subject_ID</th><th>Prediction</th></tr>");
        response.write("<tr><td>");
        for (var i = 0; i < dataf.length; i++) {
            switch (dataf[i]) {
                case ' ' :
                    response.write("</td>");
                    response.write("</tr>");
                    if (i < dataf.length - 2) {
                        response.write("<tr>");
                        response.write("<td>");
                    }
                    break;
                case '\\' :
                    response.write("</td>");
                    response.write("<td>");
                    break;
                default :
                    if (i < dataf.length - 2) {
                        response.write(dataf[i]);
                    }
            }
        }
        response.write("</table>");
        response.write("<br/>");
        response.write("Prediction finished. If there is no result, please check the form of your input.");
        response.write("<br/>");
        response.write("1: AD   0: Healthy. (Prediction close to 1 means AD.)");
        cmdlog2 = "exit";
        execSync(cmdlog2);
        response.end();
    }
    catch {
        response.write("Something wrong.\n");
        response.write("Please check your files.");
        cmdlog2 = "exit";
        execSync(cmdlog2);
        response.end();
    }
}

function uploadt(response, postData, request) {
    console.log("Request handler 'uploadt' was called.");
    var form = new formidable.IncomingForm();
    form.parse(request, function (error, fields, files) {
        console.log(files);
        // try{
        //     let SubfolderName=getCurrentDate(2);
        //     response.writeHead(200, {"Content-Type": "text/html"});
        //     response.write("The Predictions:<br/>");
        //     fs.mkdirSync("DataUpload/"+SubfolderName);
        //     fs.mkdirSync("DataUpload/"+SubfolderName+"/UNZIP/");
        //     fs.mkdirSync("Predictions/"+SubfolderName);
        //     console.log("Upload:");
        //     let newpath = "/tmp/"+files.upload.name;
        //     let finalpath = "./DataUpload/"+SubfolderName+"/"+files.upload.name;
        //     console.log("to the new path:"+newpath);
        //     fs.renameSync(files.upload.path, newpath);
        //     console.log("upload finished");
        //     console.log("to the final path:"+finalpath);
        //     fs.copyFileSync(newpath,finalpath);
        //     console.log("copy finished");
        //     // console.log(finalpath.substring(finalpath.length-3,finalpath.length-1));
        //     // if(finalpath.substring(finalpath.length-3,finalpath.length)==="zip"){
        //     //     var cmdStr = "unzip "+finalpath+" -d "+"./DataUpload/"+SubfolderName+"/UNZIP/";
        //     //
        //     // }else if(finalpath.substring(finalpath.length-2,finalpath.length)==="gz"){
        //     //     var cmdStr = "gzip -d "+finalpath;
        //     // }else{
        //     //     var cmdStr = "unzip "+finalpath+" -d "+"./DataUpload/"+SubfolderName+"/UNZIP/";
        //     // }
        //     // execSync(cmdStr);
        // }
        // catch{
        //     response.writeHead(200, {"Content-Type": "text/html"});
        //     response.write("Something wrong.\n");
        //     response.write("Please check your files.");
        //     response.end();
        // }
    });
    response.writeHead(200, {"Content-Type": "text/plain"});
    response.write("You've sent the text: " +
        querystring.parse(postData).text);
    response.end();
}

function show(response) {
    console.log("Request handler 'show' was called.");
    fs.readFile("/tmp/test.png", "binary", function (error, file) {
        if (error) {
            response.writeHead(500, {"Content-Type": "text/html"});
            response.write(error + "\n好好按步骤操作！");
            response.end();
        } else {
            response.writeHead(200, {"Content-Type": "image/png"});
            response.write(file, "binary");
            response.end();
        }
    });
}

function logo(response) {
    console.log("Request handler 'logo' was called.");
    fs.readFile("./images/index.jpg", "binary", function (error, file) {
        if (error) {
            response.writeHead(500, {"Content-Type": "text/html"});
            response.end();
        } else {
            response.writeHead(200, {"Content-Type": "image/png"});
            response.write(file, "binary");
            response.end();
        }
    });
}

function getCurrentDate(format) {
    var now = new Date();
    var year = now.getFullYear(); //得到年份
    var month = now.getMonth();//得到月份
    var date = now.getDate();//得到日期
    var day = now.getDay();//得到周几
    var hour = now.getHours();//得到小时
    var minu = now.getMinutes();//得到分钟
    var sec = now.getSeconds();//得到秒
    month = month + 1;
    if (month < 10) month = "0" + month;
    if (date < 10) date = "0" + date;
    if (hour < 10) hour = "0" + hour;
    if (minu < 10) minu = "0" + minu;
    if (sec < 10) sec = "0" + sec;
    var time = "";
    //精确到天
    if (format == 1) {
        time = year + "-" + month + "-" + date;
    }
    //精确到分
    else if (format == 2) {
        time = year + "_" + month + "_" + date + "_" + hour + "_" + minu + "_" + sec;
    }
    return time;
}

exports.start = start;
exports.disease = disease;
exports.upload = upload;
exports.upload2 = upload2;
exports.email3 = email3;
exports.email4 = email4;
exports.email2 = email2;
exports.email = email;
exports.show = show;
exports.logo = logo;
exports.uploadt = uploadt;
