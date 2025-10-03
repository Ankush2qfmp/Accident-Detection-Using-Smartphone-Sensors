/**
 * Simple EJS adapter for serving HTML files with Express
 * This allows us to use our existing HTML templates with Express
 */

const fs = require('fs');
const path = require('path');

module.exports = {
  renderFile: (filePath, options, callback) => {
    fs.readFile(filePath, 'utf8', (err, content) => {
      if (err) return callback(err);
      
      // Replace template variables with their values
      if (options) {
        Object.keys(options).forEach(key => {
          if (typeof options[key] === 'string') {
            const regex = new RegExp(`{{ ${key} }}`, 'g');
            content = content.replace(regex, options[key]);
          }
        });
      }
      
      return callback(null, content);
    });
  }
};
