"""Soft Actor-Critic losses.

See: https://arxiv.org/pdf/1812.05905.pdf
"""
from typing import Any

from sac import types
from sac import networks as sac_networks
from sac.types import Params
from sac.types import PRNGKey
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm

Transition = types.Transition


def cholesky_stable(cov, delta_abs=1e-8, delta_rel=1e-5, rho_max=0.9):
    a = cov[0, 0]
    b = cov[0, 1]
    c = cov[1, 1]
    tr = a + c
    delta = delta_abs + delta_rel * jnp.maximum(tr, 0.0)

    sigma1 = jnp.sqrt(a + delta)
    sigma2 = jnp.sqrt(c + delta)

    r = b / (sigma1 * sigma2)
    rho = rho_max * jnp.tanh(r / rho_max)
    one_minus_rho2 = jnp.maximum(1.0 - rho * rho, 0.0)

    L = jnp.array([
        [sigma1, 0.0],
        [rho * sigma2, sigma2 * jnp.sqrt(one_minus_rho2)],
    ])

    return L


def stable_cholesky_3d(
    C,
    alpha=0.05,
    delta_abs=1e-6,
    delta_rel=1e-6,
    dtype=jnp.float64,
):
    d = jnp.diag(C)
    tr = jnp.sum(d)

    delta = delta_abs + delta_rel * jnp.maximum(tr, 0.0)

    Sigma = (1.0 - alpha) * C + alpha * jnp.diag(d) + delta * jnp.eye(3, dtype=dtype)
    L = jnp.linalg.cholesky(Sigma)

    return L


def epistemic_steiner(vertices_batch, w):
    def process(vertices):
        X = vertices  # shape: (dim, n)
        centroid = jnp.mean(X, 1)
        X_centered = X - centroid[:,None]
        cov = (X_centered @ X_centered.T) / X.shape[-1]
        cov = 0.5* (cov + cov.T)

        L = cholesky_stable(cov)
        intersection_point = centroid - jnp.matmul(L, w) / jnp.sqrt(2)

        # intersection_point = jnp.clip(intersection_point, min=0.)

        return intersection_point

    return jax.vmap(process)(vertices_batch)


def cholesky_2d(cov):
    sigma11 = cov[0, 0]
    sigma12 = cov[0, 1]
    sigma22 = cov[1, 1]

    l11 = jnp.sqrt(jnp.maximum(sigma11, 1e-6))
    l21 = sigma12 / l11
    l22 = jnp.sqrt(jnp.maximum(sigma22 - l21 ** 2, 1e-6))

    return jnp.array([[l11, 0.0],
                      [l21, l22]])


def steiner(vertices_batch, w, dim=2):
    def process_triangle(vertices):
        X = vertices  # shape: (dim, n)
        centroid = jnp.mean(X, 1)
        X_centered = X - centroid[:,None]
        cov = (X_centered @ X_centered.T) #/dim
        Sigma_raw = cov/(dim+1)
        Sigma_raw += 1e-4*jnp.eye(dim, dtype=Sigma_raw.dtype)

        L = cholesky_2d(Sigma_raw)

        move_vector = w / (jnp.linalg.norm(w) + 1e-5)
        intersection_point = centroid - L @ move_vector

        return intersection_point

    return jax.vmap(process_triangle)(vertices_batch)


def steiner_diag(vertices_batch, w, dim=2):
    def process_triangle(vertices):
        X = vertices  # shape: (dim, n)
        centroid = jnp.mean(X, 1)
        X_centered = X - centroid[:,None]
        cov = (X_centered @ X_centered.T) #/dim
        Sigma_raw = cov/(dim+1)
        Sigma_raw += 1e-4*jnp.eye(dim, dtype=Sigma_raw.dtype)

        L = cholesky_2d(Sigma_raw)

        return L

    chol = jax.vmap(process_triangle)(vertices_batch)
    move_vector = w / (jnp.linalg.norm(w, axis=-1,keepdims=True) + 1e-5)
    #print(chol.shape, move_vector.shape)
    intersection_point = vertices_batch.mean(-1) - jnp.einsum('bij,bj->bi', chol, move_vector)
    return intersection_point

def make_losses(sac_network: sac_networks.SACNetworks, 
                discounting: float, 
                action_size: int, 
                budget: float,
                method: str, 
                cost_limit,
                convex,
                beta,
                redq: bool,
                task_name: str,
                ):
  """Creates the SAC losses."""
  target_entropy = -1.*action_size
  policy_network = sac_network.policy_network
  q_network = sac_network.q_network
  parametric_action_distribution = sac_network.parametric_action_distribution


  def alpha_loss(log_alpha: jnp.ndarray, policy_params: Params,
                 normalizer_params: Any, transitions: Transition,
                 key: PRNGKey) -> jnp.ndarray:
    """Eq 18 from https://arxiv.org/pdf/1812.05905.pdf."""
    dist_params = policy_network.apply(normalizer_params, policy_params,
                                       transitions.observation)
    
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key)
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    alpha = jnp.exp(log_alpha)
    alpha_loss = alpha * jax.lax.stop_gradient(-log_prob - target_entropy)
    return jnp.mean(alpha_loss)


  def critic_loss(q_params: Params, policy_params: Params,
                  normalizer_params: Any, target_q_params: Params,
                  alpha: jnp.ndarray, transitions: Transition, multiplier,
                  key: PRNGKey
                  ) -> jnp.ndarray:
    
    policy_key, sample_key, bootstrap_key = jax.random.split(key, 3)
    q_old_action = q_network.apply(normalizer_params, q_params,
                                   transitions.observation, transitions.action)

    next_dist_params = policy_network.apply(normalizer_params, policy_params,
                                            transitions.next_observation)

    next_action = parametric_action_distribution.sample_no_postprocessing(
        next_dist_params, policy_key)
    next_log_prob = parametric_action_distribution.log_prob(
        next_dist_params, next_action)
    next_action = parametric_action_distribution.postprocess(next_action)

    next_q = q_network.apply(normalizer_params, target_q_params,
                             transitions.next_observation, next_action)

    entropy_term = jnp.stack([jnp.zeros_like(next_log_prob), alpha * next_log_prob], -1)

    if redq:
        N = next_q.shape[-1]
        if method == 'cop':
            k = 3
        else:
            k = 2
        sample = jax.random.choice(sample_key, N, shape=(k,), replace=False)
        next_q = next_q[..., sample]

    if method == 'saclag':
        assert task_name == 'crl'

        q_prime = jnp.stack([next_q[:,0, 0], next_q[:,1].min(-1)], -1)

        target_q = jax.lax.stop_gradient(transitions.multi_reward +
                                transitions.discount[...,None] * discounting * 
                                (q_prime-entropy_term)) # (B, 2)
        
        truncation = transitions.extras['state_extras']['truncation']
        
        q_error_1 = jnp.square(q_old_action[:,1] - target_q[:, 1, None])
        q_error_1 *= (1 - truncation)[..., None]

        q_error_2 = jnp.square(q_old_action[:, 0, 0] - target_q[:, 0])# (B, nc)
        q_error_2 *= (1 - truncation)

        obj_loss = q_error_1.mean() + q_error_2.mean()
        
        return obj_loss

    if method in ['worst_of_both', 'saclag_ucb']:
        
        if task_name == 'lrl':
            q_min = jnp.stack([next_q[:,0].min(-1), next_q[:,1].min(-1)], -1)
        if task_name == 'crl':
            q_min = jnp.stack([next_q[:,0].max(-1), next_q[:,1].min(-1)], -1)

        target_q = jax.lax.stop_gradient(transitions.multi_reward +
                                transitions.discount[...,None] * discounting * 
                                (q_min-entropy_term)) # (B, 2)
        
        q_loss = jnp.square(q_old_action - target_q[...,None]).mean(-1)
        truncation = transitions.extras['state_extras']['truncation']
        q_loss *= (1 - truncation)[...,None]
        return q_loss.mean()
    
    if method in ['scalar']:
        assert task_name == 'lrl'
        q_sum = jnp.sum(next_q, axis=1, keepdims=True)
        q_min = jnp.where(q_sum[...,0]<q_sum[...,1], next_q[...,0], next_q[...,1])

        target_q = jax.lax.stop_gradient(transitions.multi_reward +
                                transitions.discount[...,None] * discounting * 
                                (q_min-entropy_term)) # (B, 2)
        
        q_loss = jnp.square(q_old_action - target_q[...,None]).mean(-1)
        truncation = transitions.extras['state_extras']['truncation']
        q_loss *= (1 - truncation)[...,None]
        return q_loss.mean()
    
    if method == 'cal':

        assert task_name == 'crl'

        q_cost = jnp.mean(next_q[:,0, :4], -1) + 0.5*jnp.std(next_q[:,0, :4], -1)
        
        q_min = jnp.stack([q_cost, next_q[:,1,:2].min(-1)], -1)

        target_q = jax.lax.stop_gradient(transitions.multi_reward +
                                transitions.discount[...,None] * discounting * 
                                (q_min-entropy_term)) # (B, 2)
        
        q_loss = jnp.square(q_old_action - target_q[...,None])
        truncation = transitions.extras['state_extras']['truncation']
        q_loss *= (1 - truncation)[...,None, None]
        return q_loss.mean()
    
    
    if method == 'cop':
        if task_name == 'lrl':
            # gt = jax.nn.sigmoid((next_q[:,0].mean(-1)/100.-0.8)/0.2)
            # gt = (gt - jax.nn.sigmoid(-0.8/0.2))/(jax.nn.sigmoid(1.)-jax.nn.sigmoid(-0.8/0.2))
            # w = jnp.array([1., gt.mean()])
            w = jnp.array([1., 1.])
        
        if task_name == 'crl':
            lam = jnp.exp(multiplier)
            rect = jnp.clip(convex * (cost_limit - next_q[:,0].mean(-1)), max=lam)
            gt = jax.lax.stop_gradient(lam - rect)
            w = jnp.array([-1., 1.])


        q_min = epistemic_steiner(next_q, jax.lax.stop_gradient(w))

        # w = jnp.array([-lam, 1])/jnp.sqrt(lam**2 + 1.)
        # epistemic_vector = epistemic_steiner(next_q, w) # (b, 2)

        # q_min = next_q.mean(-1) - beta*epistemic_vector

        target_q = jax.lax.stop_gradient(transitions.multi_reward +
                                transitions.discount[...,None] * discounting * 
                                (q_min-entropy_term)) # (B, 2)
        
        batch_size, _, num_pairs = next_q.shape
        #mask = jax.random.poisson(bootstrap_key, lam=0.2, shape=(batch_size, num_pairs))
        #mask = jax.random.uniform(bootstrap_key, minval=0.5, maxval=1.5, shape=(batch_size, num_pairs))
        #mask = jnp.minimum(mask, 3)
        
        q_loss = jnp.square(q_old_action - target_q[...,None])
        truncation = transitions.extras['state_extras']['truncation']
        q_loss *= (1 - truncation)[...,None, None] #*mask[:, None, :]

        # q_loss  = q_loss / (jnp.sum(mask, axis=0, keepdims=True) + 1e-6)*batch_size

        return q_loss.mean()
    
    if method == 'cop-q':
        lam = jnp.exp(multiplier)
        q_cost_mean = steiner(next_q, jnp.array([lam, 1.]))
        rect = jnp.clip(convex * (cost_limit + q_cost_mean[:,0]), max=lam)
        coeff = jax.lax.stop_gradient(lam - rect)

        # next_v = steiner(next_q, jnp.array([1., 1.]))
        next_v = steiner_diag(next_q, jnp.stack([coeff, jnp.ones_like(coeff)], -1))
        next_v_reward = next_v[:,1] - alpha * next_log_prob
        next_v_cost = next_v[:,0]

        target_q_reward = jax.lax.stop_gradient(transitions.multi_reward[...,1] +
                                    (transitions.discount * discounting) *
                                    next_v_reward) # (B, 2)

        target_q_cost = jax.lax.stop_gradient(transitions.multi_reward[...,0] +
                                    (transitions.discount * discounting) *
                                    next_v_cost) # (B, 2)

        truncation = transitions.extras['state_extras']['truncation']

        batch_size, _, num_pairs = next_q.shape
        mask = jax.random.bernoulli(bootstrap_key, shape=(batch_size, num_pairs))

        q_error_1 = q_old_action[:,1] - target_q_reward[..., None] # (B, nc)
        q_error_1 *= (1 - truncation)[..., None]*mask
        #q_error_1 = mask*q_error_1/jnp.sum(mask, 0)*256

        q_error_2 = q_old_action[:,0] - target_q_cost[..., None] # (B, nc)
        q_error_2 *= (1 - truncation)[..., None]*mask
        #q_error_2 = mask*q_error_2/jnp.sum(mask, 0)*256

        obj_loss = jnp.square(q_error_1).mean() + jnp.square(q_error_2).mean()
        return obj_loss


  def actor_loss(policy_params: Params, normalizer_params: Any,
                 q_params: Params, multiplier: Params, 
                 alpha: jnp.ndarray, transitions: Transition,
                 key: PRNGKey) -> jnp.ndarray:
    
    dist_params = policy_network.apply(normalizer_params, policy_params,
                                       transitions.observation)
    action = parametric_action_distribution.sample_no_postprocessing(
        dist_params, key)
    log_prob = parametric_action_distribution.log_prob(dist_params, action)
    action = parametric_action_distribution.postprocess(action)

    q_action = q_network.apply(normalizer_params, q_params,
                               transitions.observation, action)
    

    if task_name == 'crl':
        current_q = q_network.apply(normalizer_params, q_params,
                                transitions.observation, transitions.action)
    
    if redq:
        actor_loss = alpha * log_prob - q_action.mean(-1).sum(-1)

    # if task_name == 'lrl':
    #     pass
        # gt = jax.nn.sigmoid((current_q[:,0].mean(-1)/100.-0.8)/0.2)
        #gt = (jax.lax.stop_gradient(gt) - jax.nn.sigmoid(-0.8/0.2))/(jax.nn.sigmoid(1.)-jax.nn.sigmoid(-0.8/0.2))
        #gt = gt.mean()

    if task_name == 'crl':
        lam = jnp.exp(multiplier)
        rect = jnp.clip(convex * (cost_limit - current_q[:,0].mean(-1)), max=lam)
        gt = jax.lax.stop_gradient(lam - rect)
    
    if method == 'saclag':
        assert task_name == 'crl'
        lam = jnp.exp(multiplier)
        q_cost = q_action[:, 0, 0]
        q_reward = q_action[:,1].min(-1)

        actor_loss = alpha * log_prob - q_reward + lam*q_cost

        return jnp.mean(actor_loss), jax.lax.stop_gradient(current_q[:, 0, 0]).mean()

    if method in ['worst_of_both', 'saclag_ucb']:
        if task_name == 'lrl':
            q_cost, q_reward = q_action[:,0].min(-1), q_action[:,1].min(-1)
            #q_current = current_q[:,0].min(-1)
            actor_loss = alpha * log_prob - q_reward - q_cost
        if task_name == 'crl':
            q_cost, q_reward = q_action[:,0].max(-1), q_action[:,1].min(-1)
            q_current = current_q[:,0].max(-1)
            actor_loss = alpha * log_prob - q_reward + lam*q_cost

        return jnp.mean(actor_loss), jax.lax.stop_gradient(q_cost).mean()
    
    if method in ['scalar']:
        assert task_name == 'lrl'
        q_sum = jnp.sum(q_action, axis=1, keepdims=True)
        q_min = jnp.where(q_sum[...,0]<q_sum[...,1], q_action[...,0], q_action[...,1])

        actor_loss = alpha * log_prob - q_min[:,0] - q_min[:,1]
        return jnp.mean(actor_loss), jax.lax.stop_gradient(q_min[:,0]).mean()
    
    if method == 'cop':
        if task_name == 'lrl':
            w = jnp.array([1., 1.])
            q_min = epistemic_steiner(q_action, jax.lax.stop_gradient(w))
            #q_current = steiner(current_q, jax.lax.stop_gradient(w))[:,0]
            actor_loss = alpha * log_prob - q_min[:,1] - q_min[:,0]
            return jnp.mean(actor_loss), jax.lax.stop_gradient(q_min[:,0]).mean()
        
        if task_name == 'crl':
            w = jnp.array([-1., 1.])
            q_min = steiner(q_action, jax.lax.stop_gradient(w))
            q_current = steiner(current_q, jax.lax.stop_gradient(w))[:,0]
            rect = jnp.clip(convex * (cost_limit- q_current.mean(-1)), max=lam)
            gt = jax.lax.stop_gradient(lam - rect)

            actor_loss = alpha * log_prob - q_min[:,1] + gt.mean()*q_min[:,0]
            return jnp.mean(actor_loss), jax.lax.stop_gradient(q_current).mean()
    
    if method=='cop-q':
        lam = jnp.exp(multiplier)
        q_all_current = steiner(current_q, jnp.array([lam, 1.]))
        q_current = -q_all_current[:,0]

        rect = jnp.clip(convex * (cost_limit - q_current), max=lam)

        coeff = jax.lax.stop_gradient(lam - rect)

        #q_all = steiner(q_action, jnp.array([1., 1.]))
        q_all = steiner_diag(q_action, jnp.stack([coeff, jnp.ones_like(coeff)], -1))
        q_cost = -q_all[:,0]
        q_reward = q_all[:,1]

        actor_loss = alpha * log_prob - q_reward + coeff*q_cost

        return jnp.mean(actor_loss), jax.lax.stop_gradient(q_current).mean()

    if method == 'cal':
        assert task_name == 'crl'

        q_cost = jnp.mean(q_action[:,0, :4], -1) + 0.5*jnp.std(q_action[:,0, :4], -1)
        
        q_reward = q_action[:,1,:2].min(-1)

        q_current = jnp.mean(current_q[:,0, :4], -1) + 0.5*jnp.std(current_q[:,0, :4], -1)

        lam = jnp.exp(multiplier)
        rect = jnp.clip(convex * (cost_limit - jnp.mean(q_current)), max=lam)
        gt = jax.lax.stop_gradient(lam - rect)

        actor_loss = alpha * log_prob - q_reward + gt*q_cost

        return jnp.mean(actor_loss), jax.lax.stop_gradient(q_current).mean()

  def penalty_loss(multiplier, cost_mean):
      lambda_loss = jnp.exp(multiplier)*(cost_limit-cost_mean)
      return lambda_loss.mean()


  def cost_loss(step_length, cost_mean):
      cost_loss = jnp.exp(step_length)*(cost_mean - budget)
      return cost_loss.mean()

  
  return alpha_loss, critic_loss, actor_loss, penalty_loss, cost_loss
