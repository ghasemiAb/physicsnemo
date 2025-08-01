# SPDX-FileCopyrightText: Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from pytest_utils import import_or_fail

from physicsnemo.models.diffusion.preconditioning import (
    EDMPrecond,
    EDMPrecondSR,
    EDMPrecondSuperResolution,
    VEPrecond_dfsr,
    VEPrecond_dfsr_cond,
)
from physicsnemo.models.module import Module


def test_EDMPrecondSuperResolution_forward():
    b, c_target, x, y = 1, 3, 8, 8
    c_cond = 4

    # Create an instance of the preconditioner
    model = EDMPrecondSuperResolution(
        img_resolution=x,
        img_in_channels=c_cond,
        img_out_channels=c_target,
        use_fp16=False,
        model_type="SongUNet",
    )

    latents = torch.ones((b, c_target, x, y))
    img_lr = torch.arange(b * c_cond * x * y).reshape((b, c_cond, x, y))
    sigma = torch.tensor([10.0])

    # Forward pass
    output = model(
        x=latents,
        img_lr=img_lr,
        sigma=sigma,
    )

    # Assert the output shape is correct
    assert output.shape == (b, c_target, x, y)


def test_EDMPrecondSuperResolution_fp16_forward():
    b, c_target, x, y = 1, 3, 8, 8
    c_cond = 4

    # Create an instance of the preconditioner
    model_fp16 = EDMPrecondSuperResolution(
        img_resolution=x,
        img_in_channels=c_cond,
        img_out_channels=c_target,
        model_type="SongUNet",
        use_fp16=True,
    )

    model_fp32 = EDMPrecondSuperResolution(
        img_resolution=x,
        img_in_channels=c_cond,
        img_out_channels=c_target,
        model_type="SongUNet",
        use_fp16=False,
    )

    latents = torch.ones((b, c_target, x, y))
    img_lr = torch.arange(b * c_cond * x * y).reshape((b, c_cond, x, y))
    sigma = torch.tensor([10.0])

    # Forward pass
    output_fp16 = model_fp16(
        x=latents,
        img_lr=img_lr,
        sigma=sigma,
    )

    output_fp32 = model_fp32(
        x=latents,
        img_lr=img_lr,
        sigma=sigma,
    )

    # Assert the output shape is correct
    assert output_fp16.shape == (b, c_target, x, y)

    # Assert the fp16 output and fp32 output are close
    assert torch.allclose(output_fp16, output_fp32, rtol=1e-3, atol=1e-3), (
        "FP16 and FP32 outputs differ more than allowed"
    )


@import_or_fail("termcolor")
def test_EDMPrecondSuperResolution_serialization(tmp_path, pytestconfig):
    from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

    module = EDMPrecondSuperResolution(8, 1, 1)
    model_path = tmp_path / "output.mdlus"
    module.save(model_path.as_posix())
    loaded = Module.from_checkpoint(model_path.as_posix())
    assert isinstance(loaded, EDMPrecondSuperResolution)
    save_checkpoint(path=tmp_path, models=module, epoch=1)
    epoch = load_checkpoint(path=tmp_path)
    assert epoch == 1


@pytest.mark.parametrize("channels", [[0, 4], [3, 8], [3, 5]])
def test_EDMPrecond_forward(channels):
    res = [32, 64]
    cond_ch, out_ch = channels
    b = 1

    # Create an instance of the preconditioner
    model = EDMPrecond(
        img_resolution=res,
        img_channels=99,  # dummy value, should be overwritten by following args
        img_in_channels=out_ch + cond_ch,
        img_out_channels=out_ch,
        model_type="SongUNet",
    )

    latents = torch.randn(b, out_ch, *res)
    sigma = torch.tensor([10.0])

    if cond_ch > 0:
        # Forward pass with conditioning
        condition = torch.randn(b, cond_ch, *res)
        output = model(
            x=latents,
            condition=condition,
            sigma=sigma,
        )
    else:
        # Forward pass without conditioning
        output = model(
            x=latents,
            sigma=sigma,
        )

    # Assert the output shape is correct
    assert output.shape == (b, out_ch, *res)


def test_VEPrecond_dfsr():
    b, c, x, y = 1, 3, 256, 256
    img_resolution = 256
    img_channels = 3
    model_kwargs = {
        "embedding_type": "positional",
        "encoder_type": "standard",
        "decoder_type": "standard",
        "channel_mult_noise": 1,
        "resample_filter": [1, 1],
        "model_channels": 64,
        "channel_mult": [1, 1, 1, 2],
        "dropout": 0.13,
    }

    preconditioned_model = VEPrecond_dfsr(
        img_resolution=img_resolution,
        img_channels=img_channels,
        label_dim=0,
        use_fp16=False,
        sigma_min=0.02,
        sigma_max=100.0,
        dataset_mean=5.85e-05,
        dataset_scale=4.79,
        model_type="SongUNet",
        **model_kwargs,
    )

    xt = torch.randn(b, c, x, y)
    t = torch.randn(b)
    pred_t = preconditioned_model(xt, t)
    assert xt.size() == pred_t.size()


def test_voriticity_residual_method():
    b, c, x, y = 1, 3, 256, 256
    img_resolution = 256
    img_channels = 3
    dataset_mean = 5.85e-05
    dataset_scale = 4.79
    model_kwargs = {
        "embedding_type": "positional",
        "encoder_type": "standard",
        "decoder_type": "standard",
        "channel_mult_noise": 1,
        "resample_filter": [1, 1],
        "model_channels": 64,
        "channel_mult": [1, 1, 1, 2],
        "dropout": 0.13,
    }

    preconditioned_model = VEPrecond_dfsr_cond(
        img_resolution=img_resolution,
        img_channels=img_channels,
        label_dim=0,
        use_fp16=False,
        sigma_min=0.02,
        sigma_max=100.0,
        dataset_mean=dataset_mean,
        dataset_scale=dataset_scale,
        model_type="SongUNet",
        **model_kwargs,
    )

    xt = torch.randn(b, c, x, y)
    dx_t = preconditioned_model.voriticity_residual(
        (xt * dataset_scale + dataset_mean) / dataset_scale
    )

    assert xt.size() == dx_t.size()


def test_EDMPrecondSuperResolution_properties():
    """Test EDMPrecondSuperResolution amp_mode and profile_mode properties"""

    res, cin, cout = 8, 1, 1
    model = EDMPrecondSuperResolution(
        img_resolution=res,
        img_in_channels=cin,
        img_out_channels=cout,
        model_type="SongUNet",
    )

    # Default value should be False
    assert model.amp_mode is False

    # Enable amp_mode and verify propagation
    model.amp_mode = True
    assert model.amp_mode is True
    if hasattr(model.model, "amp_mode"):
        assert model.model.amp_mode is True
    for sub in model.model.modules():
        if hasattr(sub, "amp_mode"):
            assert sub.amp_mode is True

    # Disable again and verify
    model.amp_mode = False
    assert model.amp_mode is False
    if hasattr(model.model, "amp_mode"):
        assert model.model.amp_mode is False
    for sub in model.model.modules():
        if hasattr(sub, "amp_mode"):
            assert sub.amp_mode is False

    # Do the same for profile_mode
    # Enable profile_mode and verify propagation
    model.profile_mode = True
    assert model.profile_mode is True
    if hasattr(model.model, "profile_mode"):
        assert model.model.profile_mode is True
    for sub in model.model.modules():
        if hasattr(sub, "profile_mode"):
            assert sub.profile_mode is True

    # Disable again and verify
    model.profile_mode = False
    assert model.profile_mode is False
    if hasattr(model.model, "profile_mode"):
        assert model.model.profile_mode is False
    for sub in model.model.modules():
        if hasattr(sub, "profile_mode"):
            assert sub.profile_mode is False


def test_EDMPrecondSR_forward():
    b, c_target, x, y = 1, 3, 8, 8
    c_cond = 4

    # Create an instance of the preconditioner
    model = EDMPrecondSR(
        img_resolution=x,
        img_channels=c_target,  # This is not used but required for backward compatibility
        img_in_channels=c_cond,
        img_out_channels=c_target,
        use_fp16=False,
        model_type="SongUNet",
    )

    latents = torch.ones((b, c_target, x, y))
    img_lr = torch.arange(b * c_cond * x * y).reshape((b, c_cond, x, y))
    sigma = torch.tensor([10.0])

    # Forward pass
    output = model(
        x=latents,
        img_lr=img_lr,
        sigma=sigma,
    )

    # Assert the output shape is correct
    assert output.shape == (b, c_target, x, y)


@import_or_fail("termcolor")
def test_EDMPrecondSR_serialization(tmp_path, pytestconfig):
    from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

    module = EDMPrecondSR(
        8, 1, 1, 1
    )  # img_resolution, img_channels, img_in_channels, img_out_channels
    model_path = tmp_path / "output.mdlus"
    module.save(model_path.as_posix())
    loaded = Module.from_checkpoint(model_path.as_posix())
    assert isinstance(loaded, EDMPrecondSR)
    save_checkpoint(path=tmp_path, models=module, epoch=1)
    epoch = load_checkpoint(path=tmp_path)
    assert epoch == 1
