// JS script for IMSE

// Author: Sayantan Hore
// Created: 01.06.2014

"use strict";


function createColorHyperlink(){
    return $("<a href></a>")
}

function createColorHyperlinkParentDiv(){
    //var link = createColorHyperlink();
    
    return $("<div class='col-xs-1 col-md-1'></div>"); 
}

function createMultipleColorHyperlinkParentDivContainer(){
    //var link = createColorHyperlink();
    
    return $("<div class='col-xs-5 col-md-5'></div>"); 
}

function createColorPaletteFragment(noOfContainers, linkVisible){
    if(noOfContainers === 1){
        var link = createColorHyperlink();
        if (linkVisible === true){
            link.addClass("thumbnail");
        }
        var linkParentDiv = createColorHyperlinkParentDiv();
        linkParentDiv.append(link);
        return createColorHyperlinkParentDiv().append(linkParentDiv);
    }
    else{
        var multipleLinkParentDiv = createMultipleColorHyperlinkParentDivContainer();
        for (var i = 0; i < noOfContainers; i++){
            var link = createColorHyperlink();
            link.addClass("thumbnail");
            var linkParentDiv = createColorHyperlinkParentDiv();
            linkParentDiv.append(link);
            multipleLinkParentDiv.append(linkParentDiv);
        }
        return multipleLinkParentDiv;
    } 
}

function colorPaletteRow(){
    var row = $("<div class = 'row'>");
    row.append(createColorPaletteFragment(1, false));
    row.append(createColorPaletteFragment(12, true));
    row.append(createColorPaletteFragment(1, true));
    row.append(createColorPaletteFragment(12, true));
    return row;
}

function createFrontPageDisplayBoxes(){

    //var sqBox = $("<div class = 'sqbox'></div>");
    //return sqBox;
    
    
    
}

$(document).ready(function(){
    //$("body").append(createFrontPageDisplayBoxes());
    for (var rowCounter = 0; rowCounter < 15; rowCounter++){
        $(".container-fluid").append(colorPaletteRow());
    }
    $(".container-fluid").append($("<div class = 'page-header'></div>"));
    //$(".container-fluid").append($("<button type='button' class='btn btn-default btn-sm pull-right'>Proceed</button>"));
});

