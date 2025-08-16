/*
 * Image.cpp
 *
 * Copyright (c) 2014-2015 SEACAVE
 *
 * Author(s):
 *
 *      cDc <cdc.seacave@gmail.com>
 *
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 * Additional Terms:
 *
 *      You are required to preserve legal notices and author attributions in
 *      that material or in the Appropriate Legal Notices displayed by works
 *      containing it.
 */

#include "Common.h"
#include "Image.h"

using namespace VIEWER;


// D E F I N E S ///////////////////////////////////////////////////


// S T R U C T S ///////////////////////////////////////////////////

Image::Image(MVS::IIndex _idx)
	:
	idx(_idx),
	texture(0)
{
}
Image::~Image()
{
	Release();
}

void Image::Release()
{
	if (IsValid()) {
		GL_CHECK(glDeleteTextures(1, &texture));
		texture = 0;
	}
	ReleaseImage();
}
void Image::ReleaseImage()
{
	if (IsImageValid()) {
		cv::Mat* const p(pImage);
		Thread::safeExchange(pImage.ptr, (int_t)IMG_NULL);
		delete p;
	}
}

void Image::SetImageLoading()
{
	ASSERT(IsImageEmpty());
	Thread::safeExchange(pImage.ptr, (int_t)IMG_LOADING);
}
void Image::AssignImage(cv::InputArray img)
{
	ASSERT(IsImageLoading());
	ImagePtrInt pImg(new cv::Mat(img.getMat()));
	if (pImg.pImage->cols%4 != 0) {
		// make sure the width is multiple of 4 (seems to be an OpenGL limitation)
		cv::resize(*pImg.pImage, *pImg.pImage, cv::Size((pImg.pImage->cols/4)*4, pImg.pImage->rows), 0, 0, cv::INTER_AREA);
	}
	Thread::safeExchange(pImage.ptr, pImg.ptr);
}
bool Image::TransferImage()
{
	if (!IsImageValid())
		return false;
	SetImage(*pImage);
	glfwPostEmptyEvent();
	ReleaseImage();
	return true;
}

void Image::SetImage(cv::InputArray img)
{
	cv::Mat image(img.getMat());
	// create texture
	GL_CHECK(glGenTextures(1, &texture));
	// select our current texture
	GL_CHECK(glBindTexture(GL_TEXTURE_2D, texture));
	// load texture
	width = image.cols;
	height = image.rows;
	ASSERT(image.channels() == 1 || image.channels() == 3);
	ASSERT(image.isContinuous());

	// Set proper internal format and pixel format
	GLenum internalFormat, pixelFormat;
	if (image.channels() == 1) {
		internalFormat = GL_R8;
		pixelFormat = GL_RED;
	} else {
		internalFormat = GL_RGB8;
		pixelFormat = GL_BGR;  // OpenCV uses BGR by default
	}

	GL_CHECK(glTexImage2D(GL_TEXTURE_2D,
				 0, internalFormat,
				 width, height,
				 0, pixelFormat,
				 GL_UNSIGNED_BYTE, image.ptr<uint8_t>()));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE));
	GL_CHECK(glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE));
}
void Image::GenerateMipmap() const {
	GL_CHECK(glBindTexture(GL_TEXTURE_2D, texture));
	GL_CHECK(glGenerateMipmap(GL_TEXTURE_2D));
}
void Image::Bind() const {
	GL_CHECK(glBindTexture(GL_TEXTURE_2D, texture));
}
/*----------------------------------------------------------------*/
