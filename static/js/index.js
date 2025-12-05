window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function() { return false; };
  image.oncontextmenu = function() { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}

// ========== Lightbox Functionality ==========
var lightboxImages = [];
var currentLightboxIndex = 0;

function initLightbox() {
  // Create lightbox HTML structure
  var lightboxHTML = `
    <div class="lightbox-overlay" id="lightbox">
      <div class="lightbox-content">
        <button class="lightbox-close" aria-label="Kapat">&times;</button>
        <button class="lightbox-nav lightbox-prev" aria-label="Önceki">&#10094;</button>
        <img class="lightbox-image" id="lightbox-img" src="" alt="">
        <button class="lightbox-nav lightbox-next" aria-label="Sonraki">&#10095;</button>
        <p class="lightbox-caption" id="lightbox-caption"></p>
        <span class="lightbox-counter" id="lightbox-counter"></span>
      </div>
    </div>
  `;
  $('body').append(lightboxHTML);

  // Collect all images
  lightboxImages = [];
  $('section img, .content img, .column img').each(function(index) {
    var $img = $(this);
    var src = $img.attr('src');
    var alt = $img.attr('alt') || '';
    var caption = $img.closest('.column, .columns').find('em').first().text() || alt;
    
    lightboxImages.push({
      src: src,
      alt: alt,
      caption: caption
    });
    
    // Add click event to image
    $img.on('click', function(e) {
      e.preventDefault();
      openLightbox(lightboxImages.indexOf(lightboxImages.filter(function(img) {
        return img.src === src;
      })[0]));
    });
  });

  // Lightbox event handlers
  $('#lightbox').on('click', function(e) {
    if (e.target === this) {
      closeLightbox();
    }
  });

  $('.lightbox-close').on('click', function(e) {
    e.stopPropagation();
    closeLightbox();
  });

  $('.lightbox-prev').on('click', function(e) {
    e.stopPropagation();
    navigateLightbox(-1);
  });

  $('.lightbox-next').on('click', function(e) {
    e.stopPropagation();
    navigateLightbox(1);
  });

  // Keyboard navigation
  $(document).on('keydown', function(e) {
    if ($('#lightbox').hasClass('active')) {
      switch(e.keyCode) {
        case 27: // Escape
          closeLightbox();
          break;
        case 37: // Left arrow
          navigateLightbox(-1);
          break;
        case 39: // Right arrow
          navigateLightbox(1);
          break;
      }
    }
  });
}

function openLightbox(index) {
  currentLightboxIndex = index;
  updateLightboxImage();
  $('#lightbox').addClass('active');
  $('body').css('overflow', 'hidden');
}

function closeLightbox() {
  $('#lightbox').removeClass('active');
  $('body').css('overflow', '');
}

function navigateLightbox(direction) {
  currentLightboxIndex += direction;
  if (currentLightboxIndex < 0) {
    currentLightboxIndex = lightboxImages.length - 1;
  } else if (currentLightboxIndex >= lightboxImages.length) {
    currentLightboxIndex = 0;
  }
  updateLightboxImage();
}

function updateLightboxImage() {
  var img = lightboxImages[currentLightboxIndex];
  $('#lightbox-img').attr('src', img.src).attr('alt', img.alt);
  $('#lightbox-caption').text(img.caption);
  $('#lightbox-counter').text((currentLightboxIndex + 1) + ' / ' + lightboxImages.length);
}


$(document).ready(function() {
    // Initialize Lightbox
    initLightbox();

    // Check for click events on the navbar burger icon
    $(".navbar-burger").click(function() {
      // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
      $(".navbar-burger").toggleClass("is-active");
      $(".navbar-menu").toggleClass("is-active");

    });

    var options = {
			slidesToScroll: 1,
			slidesToShow: 3,
			loop: true,
			infinite: true,
			autoplay: false,
			autoplaySpeed: 3000,
    }

		// Initialize all div with carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);

    // Loop on each carousel initialized
    for(var i = 0; i < carousels.length; i++) {
    	// Add listener to  event
    	carousels[i].on('before:show', state => {
    		console.log(state);
    	});
    }

    // Access to bulmaCarousel instance of an element
    var element = document.querySelector('#my-element');
    if (element && element.bulmaCarousel) {
    	// bulmaCarousel instance is available as element.bulmaCarousel
    	element.bulmaCarousel.on('before-show', function(state) {
    		console.log(state);
    	});
    }

    /*var player = document.getElementById('interpolation-video');
    player.addEventListener('loadedmetadata', function() {
      $('#interpolation-slider').on('input', function(event) {
        console.log(this.value, player.duration);
        player.currentTime = player.duration / 100 * this.value;
      })
    }, false);*/
    preloadInterpolationImages();

    $('#interpolation-slider').on('input', function(event) {
      setInterpolationImage(this.value);
    });
    setInterpolationImage(0);
    $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

    bulmaSlider.attach();

})
