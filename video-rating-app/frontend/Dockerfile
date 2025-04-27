FROM node:18-alpine

WORKDIR /app

# Copy only the package files to install dependencies
COPY package*.json ./

# Install dependencies
RUN npm install

# Now copy the full app (after deps are installed to benefit from Docker caching)
COPY . .

# Expose port (optional)
EXPOSE 5173

# Run dev server
CMD ["npm", "run", "dev", "--", "--host"]

