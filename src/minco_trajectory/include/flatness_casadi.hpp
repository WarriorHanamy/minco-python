#pragma once

#include <array>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

#include "flatness.hpp"
#include "casadi_generated/quadrotor_flatness.h"
#include "casadi_generated/quadrotor_flatness_config.hpp"

namespace minco::flatness
{

class CasadiQuadrotorFlatnessMap
{
   public:
    using ConfigType       = DefaultConfig;
    using ForwardQuery     = DefaultForwardQuery;
    using ForwardResult    = DefaultForwardResult;
    using BackwardQuery    = DefaultBackwardQuery;
    using BackwardResult   = DefaultBackwardResult;

    static constexpr bool kRuntimeConfigurable = false;

    CasadiQuadrotorFlatnessMap() : config_(casadi_generated::kEmbeddedConfig) {}

    inline void configure(const ConfigType &config)
    {
        (void)config;
        // Runtime reconfiguration is not supported. Emit a helpful error.
        throw std::logic_error(
            "CasADi flatness map embeds parameters at code generation time. "
            "Re-run tools/build_casadi_flatness to update the compiled model.");
    }

    inline void configure_from_file(const std::string &file_path = std::string())
    {
        if (!file_path.empty())
        {
            throw std::invalid_argument(
                "CasADi flatness map ignores runtime config files. "
                "Re-run tools/build_casadi_flatness to embed updated parameters.");
        }
    }

    inline const ConfigType &config() const { return config_; }

    inline ForwardResult forward(const ForwardQuery &query)
    {
        auto result = ForwardResult{};
        forward(query.velocity, query.acceleration, query.jerk, query.yaw,
                query.yaw_rate, result.thrust, result.quaternion,
                result.angular_velocity);
        return result;
    }

    inline BackwardResult backward(const BackwardQuery &query) const
    {
        BackwardResult result;
        backward(query.position_gradient, query.velocity_gradient,
                 query.thrust_gradient, query.quaternion_gradient,
                 query.angular_velocity_gradient, result.position_total_gradient,
                 result.velocity_total_gradient, result.acceleration_total_gradient,
                 result.jerk_total_gradient, result.yaw_total_gradient,
                 result.yaw_rate_total_gradient);
        return result;
    }

    inline void forward(const Eigen::Vector3d &vel, const Eigen::Vector3d &acc,
                        const Eigen::Vector3d &jer, const double &psi,
                        const double &dpsi, double &thr, Eigen::Vector4d &quat,
                        Eigen::Vector3d &omg)
    {
        const auto &sizes = forward_workspace_sizes();

        std::array<const double *, 5> args{
            vel.data(),
            acc.data(),
            jer.data(),
            &psi,
            &dpsi,
        };

        std::array<double, 8> buffer{};
        std::array<double *, 1> res{buffer.data()};

        std::vector<casadi_int> iw(static_cast<std::size_t>(sizes[2]));
        std::vector<double>     w(static_cast<std::size_t>(sizes[3]));

        check_status(casadi_quadrotor_flatness_forward(args.data(), res.data(),
                                                       iw.data(), w.data(), 0),
                     "casadi_quadrotor_flatness_forward");

        thr  = buffer[0];
        quat = Eigen::Map<const Eigen::Vector4d>(buffer.data() + 1);
        omg  = Eigen::Map<const Eigen::Vector3d>(buffer.data() + 5);

        last_forward_query_.emplace();
        last_forward_query_->velocity      = vel;
        last_forward_query_->acceleration  = acc;
        last_forward_query_->jerk          = jer;
        last_forward_query_->yaw           = psi;
        last_forward_query_->yaw_rate      = dpsi;
    }

    inline void backward(const Eigen::Vector3d &pos_grad,
                         const Eigen::Vector3d &vel_grad, const double &thr_grad,
                         const Eigen::Vector4d &quat_grad,
                         const Eigen::Vector3d &omg_grad,
                         Eigen::Vector3d &pos_total_grad,
                         Eigen::Vector3d &vel_total_grad,
                         Eigen::Vector3d &acc_total_grad,
                         Eigen::Vector3d &jer_total_grad, double &psi_total_grad,
                         double &dpsi_total_grad) const
    {
        if (!last_forward_query_)
        {
            throw std::logic_error(
                "CasADi flatness backward requires a prior forward evaluation.");
        }

        const auto &state = *last_forward_query_;
        const auto &sizes = backward_workspace_sizes();

        std::array<const double *, 10> args{
            state.velocity.data(),
            state.acceleration.data(),
            state.jerk.data(),
            &state.yaw,
            &state.yaw_rate,
            pos_grad.data(),
            vel_grad.data(),
            &thr_grad,
            quat_grad.data(),
            omg_grad.data(),
        };

        std::array<double, 14> buffer{};
        std::array<double *, 1> res{buffer.data()};

        std::vector<casadi_int> iw(static_cast<std::size_t>(sizes[2]));
        std::vector<double>     w(static_cast<std::size_t>(sizes[3]));

        check_status(casadi_quadrotor_flatness_backward(args.data(), res.data(),
                                                        iw.data(), w.data(), 0),
                     "casadi_quadrotor_flatness_backward");

        pos_total_grad = Eigen::Map<const Eigen::Vector3d>(buffer.data());
        vel_total_grad = Eigen::Map<const Eigen::Vector3d>(buffer.data() + 3);
        acc_total_grad = Eigen::Map<const Eigen::Vector3d>(buffer.data() + 6);
        jer_total_grad = Eigen::Map<const Eigen::Vector3d>(buffer.data() + 9);
        psi_total_grad = buffer[12];
        dpsi_total_grad = buffer[13];
    }

   private:
    ConfigType                               config_{};
    mutable std::optional<ForwardQuery>      last_forward_query_{};

    static inline const std::array<casadi_int, 4> &forward_workspace_sizes()
    {
        static const std::array<casadi_int, 4> sizes = [] {
            casadi_int arg = 0;
            casadi_int res = 0;
            casadi_int iw  = 0;
            casadi_int w   = 0;
            casadi_quadrotor_flatness_forward_work(&arg, &res, &iw, &w);
            return std::array<casadi_int, 4>{arg, res, iw, w};
        }();
        return sizes;
    }

    static inline const std::array<casadi_int, 4> &backward_workspace_sizes()
    {
        static const std::array<casadi_int, 4> sizes = [] {
            casadi_int arg = 0;
            casadi_int res = 0;
            casadi_int iw  = 0;
            casadi_int w   = 0;
            casadi_quadrotor_flatness_backward_work(&arg, &res, &iw, &w);
            return std::array<casadi_int, 4>{arg, res, iw, w};
        }();
        return sizes;
    }

    static inline void check_status(const int status, const char *label)
    {
        if (status != 0)
        {
            throw std::runtime_error(std::string(label) +
                                     " failed to evaluate CasADi function");
        }
    }
};

}  // namespace minco::flatness
