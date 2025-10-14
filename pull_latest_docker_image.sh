#!/bin/bash

# Script to pull the newest Docker image based on local git tags and save as tar file
# Usage: ./pull_latest_docker_image.sh [output_dir]

set -e  # Exit on any error

# Configuration
OUTPUT_DIR=${1:-"./docker_images"}  # Default output directory
REGISTRY_URL="crpi-i9bomwox41wvx0ht.cn-hangzhou.personal.cr.aliyuncs.com/kangni_agents/kangni_agents"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    if ! command_exists git; then
        print_error "git is not installed. Please install git first."
        exit 1
    fi
    
    if ! command_exists docker; then
        print_error "docker is not installed. Please install docker first."
        exit 1
    fi
    
    print_success "All prerequisites are available"
}

# Function to get the latest release tag from local git repository
get_latest_release_tag() {
    print_info "Getting latest release tag from local git repository..." >&2
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir > /dev/null 2>&1; then
        print_error "Not in a git repository. Please run this script from the project root."
        exit 1
    fi
    
    # Get all release-v** tags and sort them by version
    print_info "Finding release tags..." >&2
    LATEST_TAG=$(git tag -l "release-v*" | sort -V | tail -1)
    
    if [ -z "$LATEST_TAG" ]; then
        print_error "No release-v* tags found in the repository"
        exit 1
    fi
    
    print_success "Latest release tag found: $LATEST_TAG" >&2
    echo "$LATEST_TAG"
}

# Function to pull Docker image
pull_docker_image() {
    local release_tag=$1
    # Extract version number from release tag (e.g., release-v0.5.5 -> 0.5.5)
    local version_tag=$(echo "$release_tag" | sed 's/release-v//')
    local full_image_name="${REGISTRY_URL}:${version_tag}"
    
    print_info "Pulling Docker image: $full_image_name"
    
    if docker pull "$full_image_name"; then
        print_success "Successfully pulled image: $full_image_name"
    else
        print_error "Failed to pull image: $full_image_name"
        exit 1
    fi
}

# Function to save Docker image as tar file
save_docker_image() {
    local release_tag=$1
    # Extract version number from release tag (e.g., release-v0.5.5 -> 0.5.5)
    local version_tag=$(echo "$release_tag" | sed 's/release-v//')
    local full_image_name="${REGISTRY_URL}:${version_tag}"
    local output_file="${OUTPUT_DIR}/kangni_agents-${version_tag}.tar"
    
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_DIR"
    
    print_info "Saving Docker image to: $output_file"
    
    if docker save -o "$output_file" "$full_image_name"; then
        print_success "Successfully saved image to: $output_file"
        
        # Show file size
        local file_size=$(du -h "$output_file" | cut -f1)
        print_info "File size: $file_size"
    else
        print_error "Failed to save image to: $output_file"
        exit 1
    fi
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [output_dir]"
    echo ""
    echo "Arguments:"
    echo "  output_dir  Output directory for tar file (default: ./docker_images)"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Use default output directory"
    echo "  $0 /path/to/output                   # Custom output directory"
    echo ""
    echo "Description:"
    echo "  This script pulls the latest Docker image from Alibaba Cloud Container Registry"
    echo "  based on the newest 'release-v*' git tag from the local repository."
    echo "  The image will be saved as a tar file in the specified output directory."
}

# Main function
main() {
    # Check for help flag
    if [[ "$1" == "-h" || "$1" == "--help" ]]; then
        show_usage
        exit 0
    fi
    
    print_info "Starting Docker image pull process..."
    print_info "Registry URL: $REGISTRY_URL"
    print_info "Output directory: $OUTPUT_DIR"
    
    # Check prerequisites
    check_prerequisites
    
    # Get the latest release tag
    LATEST_TAG=$(get_latest_release_tag)
    
    # Pull the Docker image
    pull_docker_image "$LATEST_TAG"
    
    # Save the Docker image as tar file
    save_docker_image "$LATEST_TAG"
    
    # Extract version number for final output
    local version_tag=$(echo "$LATEST_TAG" | sed 's/release-v//')
    
    print_success "Process completed successfully!"
    print_info "Latest image: ${REGISTRY_URL}:${version_tag}"
    print_info "Saved to: ${OUTPUT_DIR}/kangni_agents-${version_tag}.tar"
}

# Run main function
main "$@"
