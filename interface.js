function getby(x) {
  readfilewithinput(x);
}


function readfilewithinput(x) {
 var y=0;
  var file=document.getElementById("csvfile").files[0];
  var reader=new FileReader();
  reader.readAsText(file);
 reader.onload=function(){
   var getdata=[];
   lines=this.result.split("\n");
    
    var showline='<table border="1">';
    showline+='<tr>';
    for (var i=1;i<lines[0].split(',').length;i++){
     if (i<4){showline+='<td>';}
      if (i<4){showline+=lines[0].split(',')[i];}
      if (lines[0].split(',')[i]==x){
       y=i;
      }
      if (i<4){showline+='</td>';}
    }
  showline+='</tr>'
    
    
   for (var i=0;i<lines.length;i++){
       //following awesome regexp is cited from : https://stackoverflow.com/questions/11456850/split-a-string-by-commas-but-ignore-commas-within-double-quotes-using-javascript 
      line=lines[i].split(/,(?=(?:(?:[^"]*"){2})*[^"]*$)/);
      
      if(line[y]==1){
       getdata.push(line);
      }
    }
   getdata.sort(function(a,b){return b[3]<a[3];})
    for (var i=0;i<getdata.length;i++){
     showline+='<tr>';
      for (var j=1;j<4;j++){
       showline+='<td>';
        showline+=getdata[i][j];
        showline+='</td>';
      }
      showline+='</tr>'
    }
    showline+='</table>'
   document.getElementById('showbar').innerHTML =showline;
  }
  
}