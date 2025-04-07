/**
 * WDBX Documentation - Custom JavaScript
 */

document.addEventListener('DOMContentLoaded', function() {
  // Add "Copy" buttons to code blocks
  addCopyButtons();
  
  // Add heading anchors for easy linking
  addHeadingAnchors();
  
  // Add version selector if multiple versions exist
  setupVersionSelector();
  
  // Add table of contents highlighting based on scroll position
  setupTOCHighlighting();
  
  console.log('WDBX documentation custom JS initialized');
});

/**
 * Adds copy buttons to all code blocks
 */
function addCopyButtons() {
  // Find all the code blocks
  const codeBlocks = document.querySelectorAll('pre');
  
  codeBlocks.forEach(function(codeBlock) {
    // Only add button if not already present
    if (!codeBlock.querySelector('.copy-button')) {
      // Create the copy button
      const button = document.createElement('button');
      button.className = 'copy-button';
      button.innerHTML = '<i class="fas fa-copy"></i>';
      button.title = 'Copy to clipboard';
      
      // Add the button to the code block
      codeBlock.appendChild(button);
      
      // Add click event
      button.addEventListener('click', function() {
        const code = codeBlock.querySelector('code') || codeBlock;
        const textToCopy = code.textContent;
        
        // Copy to clipboard
        navigator.clipboard.writeText(textToCopy).then(function() {
          // Success feedback
          button.innerHTML = '<i class="fas fa-check"></i>';
          button.classList.add('success');
          
          // Reset after 2 seconds
          setTimeout(function() {
            button.innerHTML = '<i class="fas fa-copy"></i>';
            button.classList.remove('success');
          }, 2000);
        }).catch(function(err) {
          console.error('Could not copy text: ', err);
          button.innerHTML = '<i class="fas fa-times"></i>';
          button.classList.add('error');
          
          setTimeout(function() {
            button.innerHTML = '<i class="fas fa-copy"></i>';
            button.classList.remove('error');
          }, 2000);
        });
      });
    }
  });
}

/**
 * Adds anchor links to headings
 */
function addHeadingAnchors() {
  const headings = document.querySelectorAll('h2, h3, h4, h5, h6');
  
  headings.forEach(function(heading) {
    if (heading.id && !heading.querySelector('.heading-anchor')) {
      const anchor = document.createElement('a');
      anchor.className = 'heading-anchor';
      anchor.innerHTML = '<i class="fas fa-link"></i>';
      anchor.href = '#' + heading.id;
      anchor.title = 'Link to this heading';
      
      heading.appendChild(anchor);
    }
  });
}

/**
 * Sets up version selector if multiple versions exist
 */
function setupVersionSelector() {
  const versionSelector = document.querySelector('.version-selector');
  
  if (versionSelector) {
    versionSelector.addEventListener('change', function(e) {
      window.location.href = e.target.value;
    });
  }
}

/**
 * Highlights current section in table of contents based on scroll position
 */
function setupTOCHighlighting() {
  const toc = document.querySelector('.bd-toc-nav');
  
  if (!toc) return;
  
  const tocLinks = toc.querySelectorAll('a');
  const sections = [];
  
  // Collect all sections referenced in the TOC
  tocLinks.forEach(function(link) {
    const href = link.getAttribute('href');
    if (href && href.startsWith('#')) {
      const section = document.querySelector(href);
      if (section) {
        sections.push({
          id: href.substring(1),
          element: section,
          link: link
        });
      }
    }
  });
  
  // Update active section on scroll
  function updateActiveTOC() {
    // Get current scroll position
    const scrollPosition = window.scrollY;
    
    // Find the current section
    let currentSection = null;
    
    sections.forEach(function(section) {
      const sectionTop = section.element.offsetTop - 100; // 100px offset for header
      
      if (scrollPosition >= sectionTop) {
        currentSection = section;
      }
    });
    
    // Update active class
    if (currentSection) {
      tocLinks.forEach(function(link) {
        link.classList.remove('active');
      });
      
      currentSection.link.classList.add('active');
    }
  }
  
  // Add scroll event listener
  window.addEventListener('scroll', updateActiveTOC);
  
  // Initial update
  updateActiveTOC();
} 