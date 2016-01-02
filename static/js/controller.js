"use strict";

var IMG_GAL = IMG_GAL || {};

IMG_GAL.controller = {
    firstround: function(params){
        console.log(params.loc);
        $.get(params.loc + "/firstround", {no_of_images: params.numberOfImagesPerIteration}).done(function(data){
            console.log(typeof data)
            console.log(data)
            data = data.results;
            for (var i = 0; i < data.length; i ++){
                IMG_GAL.globals.addNewImage(IMG_GAL.model.image({
                    index: data[i]
                }));
            }
            
            params.renderGallery();
        });
    },
    
    predict: function(params){
        $.get(params.loc + "/predict", {
		no_of_images: params.numberOfImagesPerIteration,
		finished: 'false',
		feedback_indices: params.feedbacked_image_indices,
		feedback: params.feedback
	    }).done(function(data){
            console.log(data.results)
            data = data.results;
            IMG_GAL.globals.resetImages();
            for (var i = 0; i < data.length; i ++){
                IMG_GAL.globals.addNewImage(IMG_GAL.model.image({
                    index: data[i]
                }));
            }
            params.deRenderGallery();
            params.renderGallery();
        });
    },

    finished: function(params){
	$.get(params.loc + "/predict", {
		finished: 'true'
	}).done(function(){
		params.deRenderGallery();
	});
    }
};
