module type Activation = sig
  val relu : float -> float
  (** [relu x] returns [x] if [x] is greater than 0, otherwise returns 0. *)

  val relu_derivative : float -> float
  (** [relu_derivative x] returns 1 if [x] is greater than 0, otherwise returns
      0. *)

  val sigmoid : float -> float
  (** [sigmoid x] returns the sigmoid of [x], calculated as
      [1.0 / (1.0 + exp (-.x))]. *)

  val sigmoid_derivative : float -> float
  (** [sigmoid_derivative x] returns the derivative of the sigmoid function at
      [x]. *)
end

module ActivationImpl : Activation = struct
  (** [relu x] returns [x] if [x] is greater than 0, otherwise returns 0. *)
  let relu x = if x > 0.0 then x else 0.0

  (** [relu_derivative x] returns 1 if [x] is greater than 0, otherwise returns
      0. *)
  let relu_derivative x = if x > 0.0 then 1.0 else 0.0

  (** [sigmoid x] returns the sigmoid of [x], calculated as
      [1.0 / (1.0 + exp (-.x))]. *)
  let sigmoid x = 1.0 /. (1.0 +. exp (-.x))

  (** [sigmoid_derivative x] returns the derivative of the sigmoid function at
      [x]. *)
  let sigmoid_derivative x =
    let s = sigmoid x in
    s *. (1.0 -. s)
end

module type Neuron = sig
  type t
  (** [t] is a neuron that contains weights and a bias for a single neuron. *)

  type input = float list
  (** [input] is a list of floating point values representing the inputs to the
      neuron. *)

  type output = float list
  (** [output] is a list of floating point values representing the output from
      the neuron. *)

  type weight = float
  (** [weight] is a floating point value representing a weight of the neuron. *)

  val create : int -> t
  (** [create n] creates a neuron with [n] random weights and a random bias. *)

  val forward : t -> input -> output
  (** [forward neuron input] computes the output of the neuron for the given
      [input]. *)

  val update_weights : t -> weight list -> t
  (** [update_weights neuron new_weights] updates the weights of the neuron to
      the new values. *)

  val loss : t -> float -> input -> float
  (** [loss neuron expected input] computes the loss between the predicted
      output and the expected output. *)

  val train : t -> (input * float) list -> int -> float -> t
  (** [train neuron dataset epochs lr] trains the neuron using the dataset for
      [epochs] iterations and a learning rate of [lr]. *)

  val backward : t -> input -> float -> float list
  (** [backward neuron input expected] computes the gradients and returns the
      errors for each input. *)
end

module NeuronImpl : Neuron = struct
  type t = {
    weights : float list;
    bias : float;
  }
  (** [t] represents a neuron with a list of weights and a bias. *)

  type input = float list
  (** [input] is a list of floating point values representing the inputs to the
      neuron. *)

  type output = float list
  (** [output] is a list of floating point values representing the output from
      the neuron. *)

  type weight = float
  (** [weight] is a floating point value representing a weight of the neuron. *)

  (** [create n] creates a neuron with [n] random weights and a random bias. *)
  let create n =
    {
      weights = List.init n (fun _ -> Random.float 1.0);
      bias = Random.float 1.0;
    }

  (** [forward neuron input] computes the output of the neuron for the given
      [input]. *)
  let forward neuron input =
    let weighted_sum =
      List.map2 ( *. ) neuron.weights input |> List.fold_left ( +. ) 0.0
    in
    [ ActivationImpl.sigmoid (weighted_sum +. neuron.bias) ]

  (** [update_weights neuron new_weights] updates the weights of the neuron to
      the new values. *)
  let update_weights neuron new_weights = { neuron with weights = new_weights }

  (** [loss neuron expected input] computes the loss between the predicted
      output and the expected output. *)
  let loss neuron expected input =
    let prediction = forward neuron input |> List.hd in
    0.5 *. ((prediction -. expected) ** 2.0)

  (** [backward neuron input expected] computes the gradients and returns the
      errors for each input. *)
  let backward neuron input expected =
    let prediction = forward neuron input |> List.hd in
    let error =
      (prediction -. expected) *. ActivationImpl.sigmoid_derivative prediction
    in
    List.map (fun x -> error *. x) input

  (** [train neuron dataset epochs lr] trains the neuron using the dataset for
      [epochs] iterations and a learning rate of [lr]. *)
  let train neuron dataset epochs lr =
    let rec train_epoch n neuron =
      if n = 0 then neuron
      else
        let updated_neuron =
          List.fold_left
            (fun acc (x, y) ->
              let gradients = backward acc x y in
              let new_weights =
                List.map2 (fun w g -> w -. (lr *. g)) acc.weights gradients
              in
              {
                weights = new_weights;
                bias = acc.bias -. (lr *. List.hd gradients);
              })
            neuron dataset
        in
        train_epoch (n - 1) updated_neuron
    in
    train_epoch epochs neuron
end

module type Optimizer = sig
  val update : float list -> float list -> float -> float list
  (** [update weights gradients lr] updates the weights based on the gradients
      and the learning rate [lr]. *)
end

module SGD : Optimizer = struct
  (** [update weights gradients lr] updates the weights based on the gradients
      and the learning rate [lr]. *)
  let update weights gradients lr =
    List.map2 (fun w g -> w -. (lr *. g)) weights gradients
end

module type Layer = sig
  type t
  (** [t] is a layer consisting of a list of neurons. *)

  type input = float list
  (** [input] is a list of floating point values representing the inputs to the
      layer. *)

  type output = float list
  (** [output] is a list of floating point values representing the output from
      the layer. *)

  val create : int -> int -> t
  (** [create num_neurons inputs_per_neuron] creates a layer with [num_neurons]
      neurons, each having [inputs_per_neuron] inputs. *)

  val forward : t -> input -> output
  (** [forward layer input] computes the output of the layer for the given
      [input]. *)

  val backward : t -> input -> output -> input
  (** [backward layer input output_errors] computes the gradients for the layer
      based on the output errors. *)
end

module LayerImpl : Layer = struct
  type t = NeuronImpl.t list
  (** [t] is a layer consisting of a list of neurons. *)

  type input = float list
  (** [input] is a list of floating point values representing the inputs to the
      layer. *)

  type output = float list
  (** [output] is a list of floating point values representing the output from
      the layer. *)

  (** [create num_neurons inputs_per_neuron] creates a layer with [num_neurons]
      neurons, each having [inputs_per_neuron] inputs. *)
  let create num_neurons inputs_per_neuron =
    List.init num_neurons (fun _ -> NeuronImpl.create inputs_per_neuron)

  (** [forward layer input] computes the output of the layer for the given
      [input]. *)
  let forward layer input =
    List.map (fun neuron -> List.hd (NeuronImpl.forward neuron input)) layer

  (** [backward layer input output_errors] computes the gradients for the layer
      based on the output errors. *)
  let backward layer input output_errors =
    List.fold_left2
      (fun acc neuron error ->
        let gradients = NeuronImpl.backward neuron input error in
        List.map2 ( +. ) acc gradients)
      (List.init (List.length input) (fun _ -> 0.0))
      layer output_errors
end

module type Network = sig
  type t
  (** [t] is a neural network consisting of a list of layers. *)

  type input = float list
  (** [input] is a list of floating point values representing the input data to
      the network. *)

  type output = float list
  (** [output] is a list of floating point values representing the output data
      from the network. *)

  val create : int list -> t
  (** [create layers] creates a network with layers specified by the list
      [layers]. Each element in the list represents the number of neurons in the
      corresponding layer. *)

  val forward : t -> input -> output
  (** [forward network input] computes the output of the network for the given
      [input]. *)

  val train : t -> (input * output) list -> int -> float -> t
  (** [train network dataset epochs lr] trains the network using the dataset for
      [epochs] iterations and a learning rate of [lr]. *)
end

module NetworkImpl : Network = struct
  type t = LayerImpl.t list
  (** [t] is a neural network consisting of a list of layers. *)

  type input = float list
  (** [input] is a list of floating point values representing the input data to
      the network. *)

  type output = float list
  (** [output] is a list of floating point values representing the output data
      from the network. *)

  (** [create layers] creates a network with layers specified by the list
      [layers]. Each element in the list represents the number of neurons in the
      corresponding layer. *)
  let create layers =
    match layers with
    | [] | [ _ ] ->
        failwith "A network must have at least one hidden and one output layer"
    | _ ->
        let rec build_layers = function
          | i :: j :: rest -> LayerImpl.create j i :: build_layers (j :: rest)
          | _ -> []
        in
        build_layers layers

  (** [forward network input] computes the output of the network for the given
      [input]. *)
  let forward network input =
    List.fold_left (fun acc layer -> LayerImpl.forward layer acc) input network

  (** [train network dataset epochs lr] trains the network using the dataset for
      [epochs] iterations and a learning rate of [lr]. *)
  let train network dataset epochs lr =
    let rec train_epoch n network =
      if n = 0 then network
      else
        let updated_network =
          List.fold_left
            (fun net (x, y) ->
              let predictions = forward net x in
              let errors = List.map2 ( -. ) predictions y in
              let _ =
                List.fold_right2 LayerImpl.backward net (x :: []) errors
              in
              net)
            network dataset
        in
        train_epoch (n - 1) updated_network
    in
    train_epoch epochs network
end
