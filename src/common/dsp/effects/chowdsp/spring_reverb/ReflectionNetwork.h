/*
 * Surge XT - a free and open source hybrid synthesizer,
 * built by Surge Synth Team
 *
 * Learn more at https://surge-synthesizer.github.io/
 *
 * Copyright 2018-2024, various authors, as described in the GitHub
 * transaction log.
 *
 * Surge XT is released under the GNU General Public Licence v3
 * or later (GPL-3.0-or-later). The license is found in the "LICENSE"
 * file in the root of this repository, or at
 * https://www.gnu.org/licenses/gpl-3.0.en.html
 *
 * Surge was a commercial product from 2004-2018, copyright and ownership
 * held by Claes Johanson at Vember Audio during that period.
 * Claes made Surge open source in September 2018.
 *
 * All source for Surge XT is available at
 * https://github.com/surge-synthesizer/surge
 */

#ifndef SURGE_SRC_COMMON_DSP_EFFECTS_CHOWDSP_SPRING_REVERB_REFLECTIONNETWORK_H
#define SURGE_SRC_COMMON_DSP_EFFECTS_CHOWDSP_SPRING_REVERB_REFLECTIONNETWORK_H

#include <algorithm>
#include <array>
#include <cmath>

#include "globals.h"
#include "UnitConversions.h"
#include <vembertech/portable_intrinsics.h>

#include "../shared/chowdsp_DelayLine.h"
#include "../shared/Shelf.h"

namespace chowdsp
{
class ReflectionNetwork
{
  public:
    ReflectionNetwork() = default;

    void prepare(float sampleRate, int samplesPerBlock)
    {
        fs = sampleRate;

        delay.prepare({sampleRate, (juce::uint32)samplesPerBlock, 2});

        shelfFilter.reset();
    }

    void setParams(float reverbSize, float t60, float mix, float damping)
    {
        constexpr float baseDelaysSec[4] = {0.07f, 0.17f, 0.23f, 0.29f};

        float feedbackArr alignas(16)[4];
        for (int i = 0; i < 4; ++i)
        {
            auto delaySamples = baseDelaysSec[i] * reverbSize * fs;
            taps[i] = TapDelay::makeTap(delaySamples);
            feedbackArr[i] = std::pow(0.001f, delaySamples / (t60 * fs));
            feedbackArr[i] *= 0.23f * mix * (0.735f + 0.235f * reverbSize);
        }

        feedback = _mm_load_ps(feedbackArr);

        float dampDB = -1.0f - 9.0f * damping;
        shelfFilter.calcCoefs(1.0f, (float)dB_to_linear((double)dampDB), 800.0f, fs);
    }

    inline float popSample(int ch) noexcept
    {
        float output alignas(16)[4];
        for (int i = 0; i < 4; ++i)
            output[i] = delay.readTap(ch, taps[i]);
        delay.incrementReadPointer(ch);

        auto outVec = _mm_load_ps(output);

        // householder matrix
        constexpr auto householderFactor = -2.0f / (float)4;
        const auto sumXhh = vSum(outVec) * householderFactor;
        outVec = _mm_add_ps(outVec, _mm_load1_ps(&sumXhh));

        outVec = _mm_mul_ps(outVec, feedback);
        return shelfFilter.processSample(vSum(outVec) / (float)4);
    }

    inline void pushSample(int ch, float x) noexcept { delay.pushSample(ch, x); }

  private:
    float fs = 48000.0f;

    /**
     * The four reflections are fed the same input, so they share one delay history and
     * read it at four taps. Each tap's Lagrange interpolation coefficients are computed
     * once per block in setParams, with the same arithmetic as DelayLine::popSample.
     */
    class TapDelay : public DelayLine<float, DelayLineInterpolationTypes::Lagrange3rd>
    {
      public:
        using DelayLine::DelayLine;

        struct Tap
        {
            int offset = 0;
            float fraction = 0.0f, c1 = 0.0f, c2 = 0.0f, c3 = 0.0f, c4 = 0.0f;
        };

        static Tap makeTap(float delaySamples)
        {
            Tap t;
            delaySamples = std::clamp(delaySamples, 0.0f, (float)maxDelaySamples);
            t.offset = (int)std::floor(delaySamples);
            t.fraction = delaySamples - (float)t.offset;
            if (t.offset >= 1) // as DelayInterpolationTypes::Lagrange3rd
            {
                t.fraction++;
                t.offset--;
            }
            const auto d1 = t.fraction - 1.0f, d2 = t.fraction - 2.0f, d3 = t.fraction - 3.0f;
            t.c1 = -d1 * d2 * d3 / 6.0f;
            t.c2 = d2 * d3 * 0.5f;
            t.c3 = -d1 * d3 * 0.5f;
            t.c4 = d1 * d2 / 6.0f;
            return t;
        }

        inline float readTap(int channel, const Tap &t) const noexcept
        {
            const auto *p = this->bufferData.getChannelPointer((size_t)channel) +
                            this->readPos[(size_t)channel] + t.offset;
            return p[0] * t.c1 + t.fraction * (p[1] * t.c2 + p[2] * t.c3 + p[3] * t.c4);
        }

        static constexpr int maxDelaySamples = 1 << 18;
    };

    TapDelay delay{TapDelay::maxDelaySamples};
    std::array<TapDelay::Tap, 4> taps{};

    __m128 feedback;

    chowdsp::ShelfFilter<> shelfFilter;
};
} // namespace chowdsp

#endif // SURGE_SRC_COMMON_DSP_EFFECTS_CHOWDSP_SPRING_REVERB_REFLECTIONNETWORK_H
