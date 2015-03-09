/*
function saveTextAsFile()
{
	var textToWrite = document.getElementById("inputTextToSave").value;
	var textFileAsBlob = new Blob([textToWrite], {type:'text/plain'});
	var fileNameToSaveAs = document.getElementById("inputFileNameToSaveAs").value;

	var downloadLink = document.createElement("a");
	downloadLink.download = fileNameToSaveAs;
	downloadLink.innerHTML = "Download File";
	if (window.webkitURL != null)
	{
		// Chrome allows the link to be clicked
		// without actually adding it to the DOM.
		downloadLink.href = window.webkitURL.createObjectURL(textFileAsBlob);
	}
	else
	{
		// Firefox requires the link to be added to the DOM
		// before it can be clicked.
		downloadLink.href = window.URL.createObjectURL(textFileAsBlob);
		downloadLink.onclick = destroyClickedElement;
		downloadLink.style.display = "none";
		document.body.appendChild(downloadLink);
	}

	downloadLink.click();
}
*/

// Write log
// ----------------------------------------------------------------------------------------------------------------------------------------
function Log(timestamp, mouseLoc, imageIndex, description){
    __EVENT_ID__ +=1;
    __EVENTS__.push([__EVENT_ID__, timestamp, mouseLoc, imageIndex, description]);
}

function scrollHandler(){
    console.log("Scrolling :: " + document.body.scrollTop);
    if ((imageObjectOnFocus !== undefined) && (imageObjectOnFocus.feedbackBox !== undefined)){
        var top = imageObjectOnFocus.image.offset().top - document.body.scrollTop + 10;
        var left = imageObjectOnFocus.image.offset().left - document.body.scrollLeft + 10;
        imageObjectOnFocus.feedbackBox.css("top", top + "px");
        imageObjectOnFocus.feedbackBox.css("left", left + "px");
    }
}

function recordEventsToFile()
{
    var csvData = Papa.unparse({
        fields: ['Event ID', 'Timestamp', 'Description', 'Mouse(X-Y)', 'Image Index', 'Event Description'],
        data: __EVENTS__
    });
    alert(csvData);
	var textToWrite = csvData;
	var textFileAsBlob = new Blob([textToWrite], {type:'text/plain'});
	var fileNameToSaveAs = "events";

	var downloadLink = document.createElement("a");
	downloadLink.download = fileNameToSaveAs;
	downloadLink.innerHTML = "Download File";
	if (window.webkitURL != null)
	{
		// Chrome allows the link to be clicked
		// without actually adding it to the DOM.
		downloadLink.href = window.webkitURL.createObjectURL(textFileAsBlob);
	}
	else
	{
		// Firefox requires the link to be added to the DOM
		// before it can be clicked.
		downloadLink.href = window.URL.createObjectURL(textFileAsBlob);
		downloadLink.onclick = destroyClickedElement;
		downloadLink.style.display = "none";
		document.body.appendChild(downloadLink);
	}

	downloadLink.click();
    __EVENTS__ = [];
}

function destroyClickedElement(event)
{
	document.body.removeChild(event.target);
}