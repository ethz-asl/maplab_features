#!/usr/bin/env zsh

# Get the current directory.
script_source=$(readlink -f "$0")
script_dir=$(dirname "$script_source")

docker_ctx=${script_dir}/docker_ctx
echo "Creating build context in: ${docker_ctx}."

# Create the context we use for setting up the images.
mkdir -p "${docker_ctx}"
"$script_dir/copy_to_ctx.sh"
echo "Finished preparing context. Building images now"

# Build the docker image.
# docker build -t "maplab.features.interface" -f "${script_dir}/Dockerfile.interface" "${docker_ctx}"
docker build -t "maplab.features.superpoint" -f "${script_dir}/Dockerfile.superpoint" "${docker_ctx}"

# Clean up.
echo "Finished building the images. Cleaning up."
rm -rf "${docker_ctx}"
