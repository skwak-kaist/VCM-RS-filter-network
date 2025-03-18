#!/bin/bash

# This file is covered by the license agreement found in the file “license.txt” in the root of this project.

# install VCM-RS dependencies

set +e

echo "Retrieving VCM-RS dependencies..."

while IFS='=' read -r key value; do
  #echo "key:" $key
  #echo "value:" $value
  if [[ "$key" =~ ^\# ]] || [[ "$key" =~ ^\s*# ]]; then
    continue # # comment
  fi
  
  if [[ "$key" =~ ^\[ ]] || [[ "$key" =~ ^\s*[ ]]; then
    #echo "Section $key"
    file_md5=
    file_path=
    message=
    command=
    continue
  fi
    
  if [[ "$key" == "file_md5" ]]; then
    file_md5=$value
  fi
  if [[ "$key" == "file_path" ]]; then
    file_path=$value
    echo "File: $file_path"
    
    if [ -f "$file_path" ]; then
    
      if [ -z "$file_md5" ]; then
        echo "Local file found, no MD5 check"
        unset file_path
        continue
      elif [ $(md5sum "$file_path" | cut -f 1 -d ' ') = $file_md5 ]; then
        echo "Correct MD5"
        unset file_path
        continue
      else
        echo "Wrong MD5 - removing file"
        rm $file_path
        continue
      fi
    else
      echo "Local file not found"
      continue
    fi
  fi
  if [ -z "$file_path" ]; then # already finished and correct md5
    continue
  fi

  if [[ "$key" == "message" ]]; then
    message=$value
  fi
  if [[ "$key" == "command" ]]; then
    command=$value
    
    file_dir=$(dirname "$file_path")
    if [ ! -d  $file_dir ]; then
      mkdir -p $file_dir
      echo "Making dir: $file_dir"
    fi
    
    echo "Retrieving $message"
    eval "$command"

    if [ -f "$file_path" ]; then
    
      if [ -z "$file_md5" ]; then
        echo "Retrieved file without MD5 check"
        unset file_path
        continue
      elif [ $(md5sum "$file_path" | cut -f 1 -d ' ') = $file_md5 ]; then
        echo "Correct retrieved MD5"
        unset file_path
        continue
      else
        echo "Wrong MD5 - removing file"
        rm $file_path
        continue
      fi
    else
      echo "Failed to retrieve file"
      continue
    fi
    
  fi
  if [[ "$key" == "missing" ]]; then
    exception=$value
    echo "Totally failed to retrieve file. $exception"
    exit 1
  fi
  
done < dependencies.ini

echo "Finished retrieving VCM-RS dependencies"
echo
